#!/usr/bin/env python3


""" Predict the STEC O-serogroup and PFGE pattern from STEC whole genome sequence.
    
    Input: Pre-generated models, selected features, label encoders
    Output: STEC Serotype and PFGE prediction
    
    Authors: Sonali Gupta, Sung Im
    Last Modified: 2021-07-15

"""


import os
import time
import logging
import pickle
import random
import argparse
import subprocess

import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score


def get_args():
    """ Get cmd line arguments. """
    parser = argparse.ArgumentParser(
        description='Predict the STEC O-serogroup and PFGE pattern from STEC whole genome sequence.'
    )
    parser.add_argument(
        '-m', '--mash-exec',
        dest='mashexec',
        required=True,
        help='Absolute path to mash executable' # /data/home/mjain73/anaconda3/bin/mash
    )
    parser.add_argument(
        '-o', '--out-file',
        dest='outputfile',
        required=True,
        help='Absolute path to mash executable'
    )

startTime = time.time()

# Initialize inputs
args = get_args()

thisFile = os.path.realpath(__file__) # /User/path/pfgeBLAST/scripts/predict_pfge.py
topDirParts = thisFile.split('/')[:-1]
topDir = '/'.join(topDirParts) # /User/path/pfgeBLAST

ext = "fq"
threads = "10"
idir = os.path.join(topDir, 'InputFiles') # /data/storage/sonali/input
#outfile = "kmerTable.tsv"
PFGEmapping = os.path.join(topDir, 'MappingFiles', 'PFGE_Mapping.txt')
Seromapping = os.path.join(topDir, 'MappingFiles', 'Serotype_Mapping.txt')
Selected_kmers_Sero = os.path.join(topDir, 'SavedModels', 'SerotypeModels', 'selected_features.txt') #selected kmers location for sero prediction
Serotype_model = os.path.join(topDir, 'SavedModels', 'SerotypeModels', 'final_model') #saved serotype model
Serotype_label_encoder = os.path.join(topDir, 'SavedModels', 'SerotypeModels', 'label_encoder') #Serotype label encoder location
selected_kmers_pfge = os.path.join(topDir, 'SavedModels', 'PFGEmodels') #selected kmers for pfge prediction location
PFGE_models = os.path.join(topDir, 'SavedModels', 'PFGEmodels') #saved PFGE models folder location
PFGE_label_encoder = os.path.join(topDir, 'SavedModels', 'PFGEmodels') #PFGE label encoder location
Outfile = "/data/home/data/PFGE_BLAST/Combined/final-accuracy" #output location
mash = args.mashexec # /data/home/mjain73/anaconda3/bin/mash

# The result file
with open(Outfile, "w") as fh:
    fh.write('{}\t{}\t{}\n'.format('Sample', 'PFGE_Prediction', 'Prediction_Time')
    # fh.write("Sample" + "\t" + "PFGE_prediction" +"\t"+"Prediction Time"+"\n")

# Create dict object of PFGE Mappings
kmersDict = {}
sra2pfge = {}
sra2sero = {}
Serotypes = []

with open(PFGEmapping, "r") as f1:
    next(f1) # ignore header
    for line in f1.readlines():
        line=line.strip("\n").split("\t")
        # line[0] => SRA id
        # line[1] => PFGE pattern id
        sra2pfge[line[0]]=line[1]

# Check input directory for fasta files
#for ex in ext:
#    numFiles = int(subprocess.run("ls " + dir + "/*." + ex + " | wc -l", shell = True, stdout = subprocess.PIPE, encoding = 'utf-8').stdout.rstrip())
#    #numFiles.stdout == 0 is bad
#    if(numFiles == 0):
#        print("no valid files")
#        exit(1)

# Generate sketches from fastq files
subprocess.run("ls "+idir+"/*.fastq"+ "| sed 's/.fastq//' | xargs -I FILE -P "+threads+" sh -c 'test -f FILE.msh || "+mash+" sketch -r -m 5 -k 32 -s 10000 -o FILE FILE.fastq'", shell=True)

# Store .msh files into object
files = subprocess.run("ls "+idir+"/*.msh", shell=True, stdout=subprocess.PIPE, encoding='utf-8').stdout.rstrip().split("\n")

l2 = []
for i in files:
    l2.append(os.path.basename(i).strip('.msh'))


# Serotype prediction ----------------------------------------------#

# Create a binary matrix of kmers vs. selected kmers in model
with open(Selected_kmers_Sero, "r")as fh:
    for line in fh.readlines():
        line = line.strip("\t").split("\t")
        selected_kmers = line

# Deserealizing the MASH hashes
# kmersDict = {}
for file in l2 :
    kmersDict = {}
    kmers = subprocess.run(mash+" info -d "+idir+"/"+file+".msh"+" | grep -P '^\s+[0-9]+' | sed -r 's/\s+//g' | sed 's/,//'", shell=True, stdout=subprocess.PIPE, encoding='utf-8').stdout.rstrip().split("\n")
    selected_kmers = [float(i) for i in selected_kmers]
    kmers = [float(i) for i in kmers]

    for kmer in selected_kmers :
        if kmer in kmers:
            if kmer not in kmersDict :
                kmersDict[kmer] = {}
                kmersDict[kmer][file] = 1
            kmersDict[kmer][file] = 1
        elif kmer not in kmersDict:
            kmersDict[kmer] = {}
            kmersDict[kmer][file] = 0
        else:
            kmersDict[kmer][file] = 0

    # Convert kmersDict object into pd dataframe
    dataset1 = pd.DataFrame.from_dict(kmersDict,orient = 'index')
    dataset1 = dataset1.fillna(0)
    dataset1 = dataset1.transpose()
    dataset1 = dataset1[selected_kmers]

    f = open(Serotype_model, "rb")
    loaded = pickle.load(f)
    Y_pred = []
    Y_pred = loaded.predict(dataset1)

    with open(Serotype_label_encoder, 'rb') as f:
        labelencoder_y = pickle.load(f)

    Y_inverse = labelencoder_y.inverse_transform(Y_pred)
    Serotype_prediction = Y_inverse
    #print(Y_inverse)

    with open(selected_kmers_pfge + "/selected_features_" + Y_inverse[0] + ".txt", "r")as fh:
        for line in fh.readlines():
            line = line.strip("\t").split("\t")
            selected_kmers2= line

    # kmersDict = {}
    # Deserealize msh hash values
    kmersDict = {}
    kmers = subprocess.run(mash + " info -d " + idir + "/" + file + ".msh" + " | grep -P '^\s+[0-9]+' | sed -r 's/\s+//g' | sed 's/,//'", shell = True, stdout = subprocess.PIPE, encoding = 'utf-8').stdout.rstrip().split("\n")

    selected_kmers2 = [float(i) for i in selected_kmers2]
    kmers = [float(i) for i in kmers]

    for kmer in selected_kmers2 :
        if kmer in kmers:
            if kmer not in kmersDict :
                kmersDict[kmer] = {}
                kmersDict[kmer][file] = 1
            kmersDict[kmer][file] = 1

        elif kmer not in kmersDict:
            kmersDict[kmer] = {}
            kmersDict[kmer][file] = 0
        else:
            kmersDict[kmer][file] = 0


    # PFGE prediction ----------------------------------------------#

    # Convert kmersDict object into pd dataframe
    dataset1 = pd.DataFrame.from_dict(kmersDict,orient='index')
    dataset1 = dataset1.fillna(0)
    dataset1 = dataset1.transpose()
    dataset1 = dataset1[selected_kmers2]

    f = open(PFGE_models + "/final_model_" + Y_inverse[0], "rb")
    loaded = pickle.load(f)
    Y_pred = []
    Y_pred = loaded.predict(dataset1)

    with open(PFGE_label_encoder + "_" + Y_inverse[0], 'rb') as f:
        labelencoder_y = pickle.load(f)

    Y_inverse = labelencoder_y.inverse_transform(Y_pred)
    PFGE_prediction = Y_inverse

    with open(Outfile, "a") as fh:
        for i, j in zip(list(dataset1.index.values), Y_inverse):
            executionTime = (time.time() - startTime)
            fh.write('{}\t{}\t{}\n'.format(str(i), str(j), str(executionTime))
            # fh.write(str(i) + "\t" + str(j) +"\t"+str(executionTime)+"\n")
            