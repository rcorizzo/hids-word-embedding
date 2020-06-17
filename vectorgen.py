import os
import sys

import numpy as np

#import MoultonSplitGeneration

from gensim.corpora import Dictionary
from gensim.models import Word2Vec as W2V
from gensim.models import TfidfModel as TfIdf
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# ---- Helper Functions ----

# Input: vec1 (anysize vector), vec2 (samesize vector)
# Output: vector addition of vec1, vec2
def add_vecs(vec1, vec2):
    ret_vec = []
    for x1, x2 in zip(vec1, vec2):
        ret_vec.append(x1 + x2)
    return ret_vec


# Input: vec (anysize vector), num (scalar)
# Output: vector of each value in vec divided by num
def div_vec_by_num(vec, num):
    if num == 0:
        return vec
    ret_vec = [i for i in vec]
    return list(map(lambda x: x / num, ret_vec))


# Input: vec (anysize vector), num (scalar)
# Output: vector of each value in vec multiplied by num
def mult_vec_by_num(vec, num):
    ret_vec = [i for i in vec]
    return list(map(lambda x: x * num, ret_vec))


# Input: ls (list of items)
# Output: TaggedDocument transformation of ls. Used with Doc2Vec instead of a normal list
def list_to_tagdoc(ls):
    for i, line in enumerate(ls):
        yield TaggedDocument(line, [i])


# Input: elements (list of directories in the filepath)
# Action: creates directories in filepath
# Output: filepath string
def build_directory(elements):
    path = ''
    for element in elements:
        if len(path):
            path += '/'
        path += element
        if not os.path.exists(path):
            os.mkdir(path)
    return path


# Input: epoch (current epoch), pct_name (current datasplit), data_name (current dataset),
#        a b c (boolean of whether corresponding process is complete)
# Action: writes stylized table row to stdout
def print_update(epoch, pct_name, data_name, a, b, c):
    sys.stdout.write('\r   ' + str(epoch + 1) + '   |  ' + pct_name + ' |  ' + data_name + '\t|     '
                        + (u'\u2713' if a == 1 else (u'\u2718' if a == -1 else ' ')) + '     |     '
                        + (u'\u2713' if b == 1 else (u'\u2718' if b == -1 else ' ')) + '     |     '
                        + (u'\u2713' if c == 1 else (u'\u2718' if c == -1 else ' ')) + '     ')


# ---- Import Raw Data ----

# Input: .TXT filepath containing raw data traces
# Output: array of raw data traces
def import_raw(fpath):
    with open(fpath, 'r') as f:
        f_str = f.read()
    f.close()
    f_lines = f_str.splitlines()
    raw = [i.split() for i in f_lines]
    return raw


# Input: .ARFF filepath containing Moulton Transformed data traces
# Output: dict {'header': array of lines in the ARFF header, 'data': array of Moulton traces}
def import_moulton(fpath):
    with open(fpath, 'r') as f:
        f_str = f.read()
    f.close()
    f_lines = f_str.splitlines()
    header = f_lines[:12]
    data = f_lines[12:]
    return {'header': header, 'data': data}


# Input: normal (array of raw normal traces), attack (array of raw attack traces), epoch (current epoch)
# Output: multidimensional array of datasplits
#         25%, 50%, 100% with normal train, attack train, attack test, and normal test in each
def generate_datasets(normal, attack):
    return [normal, attack]


# ---- Build Models ----


# Input: train_corpus (raw training data)
# Output: trained Doc2Vec model
def build_d2v_model(train_corpus, vec_size):
    train_corpus = list_to_tagdoc(train_corpus)
    model = Doc2Vec(vector_size=vec_size, min_count=1, epochs=200, dm=0)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model


# Input: train_data (raw training data)
# Output: dictionary of dict (trained TF-IDF dictionary) and model (trained TF-IDF model)
def build_tfidf_model(train_data):
    tfidf_dict = Dictionary(train_data)
    tfidf_corpus = [tfidf_dict.doc2bow(trace) for trace in train_data]
    tfidf_model = TfIdf(tfidf_corpus)
    return {'dict': tfidf_dict, 'model': tfidf_model}


# Input: train_corpus (raw training data)
# Output: trained Word2Vec model
def build_w2v_model(train_corpus, vec_size):
    w2v_model = W2V(train_corpus, size=vec_size, window=5, min_count=1, workers=4)
    return w2v_model


# ---- Transform Data ----

# Input: dataset (raw trace datasplit), d2v_model (trained Doc2Vec model)
# Output: array of Doc2Vec transformed vectors
def transform_d2v(dataset, d2v_model):
    vectors = [d2v_model.infer_vector(trace.words, epochs=50) for trace in list_to_tagdoc(dataset)]
    return vectors


# Input: dataset (raw trace datasplit), w2v_model (trained Word2Vec model),
#        tfidf_model (trained TF-IDF model), tfidf_dict (trained TF-IDF dictionary)
# Output: array of Word2Vec transformed vectors with TF-IDF weighting
def transform_w2v_tfidf(dataset, w2v_model, tfidf_model, tfidf_dict):
    def trace2vec(trace):
        ret_vec = [0 for _ in range(100)]
        counter = 0

        tfidf_temp_corpus = tfidf_dict.doc2bow(trace)
        tfidf_wts = tfidf_model[tfidf_temp_corpus]

        temp_dict = {}
        for wt in tfidf_wts:
            temp_dict[wt[0]] = wt[1]

        for val in trace:
            try:
                w2v_vec = w2v_model.wv[val]
                tfidf_id = tfidf_dict.token2id[val]
                tfidf_wt = temp_dict[tfidf_id]
            except KeyError:
                continue

            ret_vec = add_vecs(ret_vec, mult_vec_by_num(w2v_vec, tfidf_wt))
            counter += 1

        ret_vec = div_vec_by_num(ret_vec, counter)
        return ret_vec

    vectors = []
    for t in dataset:
        try:
            vectors.append(trace2vec(t))
        # ZeroDivisionError is thrown when a trace contains only calls that were not represented in the training set
        except ZeroDivisionError:
            continue
    return np.array(vectors)


# Input: dataset (raw trace datasplit), w2v_model (trained Word2Vec model)
# Output: array of Word2Vec transformed vectors
def transform_w2v(dataset, w2v_model):
    def trace2vec(trace):
        ret_vec = [0 for _ in range(100)]
        counter = 0

        for val in trace:
            try:
                w2v_vec = w2v_model.wv[val]
            except KeyError:
                continue

            ret_vec = add_vecs(ret_vec, w2v_vec)
            counter += 1

        ret_vec = div_vec_by_num(ret_vec, counter)
        return ret_vec

    vectors = []
    for t in dataset:
        try:
            vectors.append(trace2vec(t))
        # Error is thrown when a trace contains only calls that were not represented in the training set
        except ZeroDivisionError:
            continue
    return np.array(vectors)

# ---- Main ----

def process_doc_word_idf(normal_data, attack_data, base_dir, it):
    
    vec_size = 512
    raw_attack = import_raw(normal_data)
    raw_normal = import_raw(attack_data)

    data_names = ['Train_Normal', 'Train_Attack', 'Test_Attack', 'Test_Normal']
    pct_names = ['25_P', '50_P', '100P']
    base_direc = base_dir + 'CompiledData'
    
    data = generate_datasets(raw_normal, raw_attack)
    #print(len(data))
    train_data = data[0] + data[1]
    w2v_model = build_w2v_model(train_data, vec_size)
    d2v_model = build_d2v_model(train_data, vec_size)
    dict_and_model = build_tfidf_model(train_data)
    tfidf_dict, tfidf_model = dict_and_model['dict'], dict_and_model['model']
    
    normal_doc2vec_fn = os.path.join(base_dir, "Doc2Vec_normal_" + str(it) + ".npy")
    attack_doc2vec_fn = os.path.join(base_dir, "Doc2Vec_attack_" + str(it) + ".npy")
    
    # Doc2Vec Transform
    d2v_data = transform_d2v(data[0], d2v_model)
    np.save(normal_doc2vec_fn, d2v_data)
    d2v_data = transform_d2v(data[1], d2v_model)
    np.save(attack_doc2vec_fn, d2v_data)
    
    normal_Word2VecAndTfIDF_fn = os.path.join(base_dir, "Word2VecAndTfIDF_normal_" + str(it) + ".npy")
    attack_Word2VecAndTfIDF_fn = os.path.join(base_dir, "Word2VecAndTfIDF_attack_" + str(it) + ".npy")    

    # Word2Vec and TF-IDF Transform
    w2v_tfidf_data = transform_w2v_tfidf(data[0], w2v_model, tfidf_model, tfidf_dict)
    np.save(normal_Word2VecAndTfIDF_fn, w2v_tfidf_data)
    w2v_tfidf_data = transform_w2v_tfidf(data[1], w2v_model, tfidf_model, tfidf_dict)
    np.save(attack_Word2VecAndTfIDF_fn, w2v_tfidf_data)

    normal_Word2Vec_fn = os.path.join(base_dir, "Word2Vec_normal_" + str(it) + ".npy")
    attack_Word2Vec_fn = os.path.join(base_dir, "Word2Vec_attack_attack_" + str(it) + ".npy")  
    
    # Word2Vec Transform
    #w2v_path = 'Word2Vec.npy'
    w2v_data = transform_w2v(data[0], w2v_model)
    np.save(normal_Word2Vec_fn, w2v_data)
    w2v_data = transform_w2v(data[1], w2v_model)
    np.save(attack_Word2Vec_fn, w2v_data)
    

"""
if __name__ == '__main__':
    main()
"""