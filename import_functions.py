# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:24:45 2020

@author: AU_CS

Import Functions
"""

# ---- Import Functions ----

import os

import random

from import_functions import  *

from build_models import *

from helper_functions import *

import csv



# Input: directory (parent directory containing raw data traces)

# Output: array of shuffled raw data traces

def import_raw(directory):

    raw = []

    for filepath in os.listdir(directory):

        with open(directory + '/' + filepath) as f:

            f_str = f.read()

        f.close()

        f_wrd = f_str.split()

        raw.append(f_wrd)

    random.shuffle(raw)

    return raw




# Imports from a designated filepath (currently hard-coded) and returns the
    
# raw_normal, raw_valid, and raw_attack variables
def import_data(filen):
   

    if os.path.exists(filen):

        with open(filen, 'r') as f:

            f_str = f.read()

            f_lines = f_str.splitlines()

            _file_complete = [i.split() for i in f_lines]

        f.close()
        
        print("IMPORT SUCCESS: ", os.path.basename(filen))

    
    return f_lines, _file_complete



# Imports from a designated filepath (currently hard-coded) and returns the
    
# raw_normal, raw_valid, and raw_attack variables
def import_csv(filen):
   

    if os.path.exists(filen):

        with open(filen, 'r') as f:

            f_str = f.read()

            f_lines = f_str.splitlines()

            _file_complete = [i.split(',') for i in f_lines]

        f.close()
        
        print("IMPORT SUCCESS: ", os.path.basename(filen))

    
    return _file_complete
