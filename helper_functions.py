# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:21:45 2020

@author: AU_CS

Helper Functions File
"""

# ---- Helper Functions ----

import os

import sys

from import_functions import  *

from build_models import *

from helper_functions import *



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

