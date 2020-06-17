# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:25:24 2020

@author: AU_CS

Build Models
"""


# ---- Build Models ----


from gensim.corpora import Dictionary

from gensim.models import Word2Vec as W2V

from gensim.models import TfidfModel as TfIdf

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


import numpy as np

import pandas as pd

from import_functions import  *

from build_models import *

from helper_functions import *

from import_functions import  *

import os


# support ngram functionality
import nltk

from nltk import ngrams, everygrams, FreqDist
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from itertools import chain

import nltk, re, string, collections



# Input: train_corpus (raw training data)

# Output: trained Doc2Vec model

def build_d2v_model(train_corpus):

    train_corpus = list_to_tagdoc(train_corpus)

    model = Doc2Vec(vector_size=128, min_count=1, epochs=200, dm=0)

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

def build_w2v_model(train_corpus):

    w2v_model = W2V(train_corpus, size=128, window=5, min_count=1, workers=4)

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

        except ZeroDivisionError:

            continue

    return np.array(vectors)





"""
Imports from a designated filepath (currently hard-coded) and returns the
just the traces
"""
def import_traces(filen):


    if os.path.exists(filen):

        with open(filen, 'r') as f:

            f_str = f.read()

            f_lines = f_str.splitlines()

        f.close()

    return f_lines


