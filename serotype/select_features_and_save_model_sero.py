#!/usr/bin/env python3

import subprocess
import argparse
import numpy as np
import pandas as pd
import time
import os
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import logging
import pickle

#############################initialising inputs########################################################

def main():

    mash = "/data/home/mjain73/anaconda3/bin/mash"
    #idir="/data/storage/sonali/sketches"
    ext="fq"
    threads="10"
    odir=""
    PFGEmapping="/data/home/data/PFGE_BLAST/Metadata/5970_PFGE_Mapping.txt"

    # script arguments
    # 1. directory => default will be current (.)
    # 2. extension => default will be FASTA
    # 3. mapping file => no default, this will be SRA-to-PFGE mapping
    # 4. threads => default = 10


    #taking user input


    parser = argparse.ArgumentParser()
    parser.add_argument("--mash")
    parser.add_argument("--idir")
    parser.add_argument("--ext")
    parser.add_argument("--threads")
    parser.add_argument("--odir")
    parser.add_argument("--idir2")
    args = parser.parse_args()

    if(args.mash):
        mash = args.mash

    if(args.idir):
        idir = args.idir

    if(args.ext):
        ext = args.ext
    if(args.threads):
        threads = args.threads
    if(args.odir):
        odir = args.odir
    if (args.idir2):
        idir2 = args.idir2




    """
    Global variables:
        1. kmersDict => 2-level dictionary with kmers (1st level) and files (2nd level)
        2. sra2pfge => SRA to PFGE mapping dictionary
    """
    kmersDict = {}
    sra2pfge = {}
    sra2sero = {}
    Serotypes = []

    global importances
    global X_model
    global Y_model

#############################Getting PFGE Mapping and Serotype Mapping into dictionary###########################################

    with open(PFGEmapping, "r") as f1:
        next(f1) # ignore header
        for line in f1.readlines():
            line=line.strip("\n").split("\t")
            # line[0] => SRA id
            # line[1] => PFGE pattern id
            sra2pfge[line[0]]=line[1]
            sra2sero[line[0]]=line[1][0:3]

    files = subprocess.run("ls " + idir + "/*.msh", shell=True, stdout=subprocess.PIPE,
                           encoding='utf-8').stdout.rstrip().split("\n")

    l2 = []
    for i in files:
        l2.append(os.path.basename(i).strip('.msh'))

    feats_dict = {}
    with open(idir2+"/selected_no_of_feats_sero", "r") as fh:
        next(fh)
        for line in fh:
            # print(line)
            line = line.rstrip().split("\t")
            feats_dict[line[0]] = line[1]
    print(feats_dict)
    for k,v in feats_dict.items():
        kmers = []
        kmersDict = {}
        for file in l2:
            kmers = subprocess.run(mash + " info -d " + idir + "/" + file + ".msh" + " | grep -P '^\s+[0-9]+' | sed -r 's/\s+//g' | sed 's/,//'",shell=True, stdout=subprocess.PIPE, encoding='utf-8').stdout.rstrip().split("\n")
            kmers = [float(i) for i in kmers]
            #print(kmers)
            for kmer in kmers:
                if kmer not in kmersDict:
                    kmersDict[kmer] = {}
                    kmersDict[kmer][file] = 1
                kmersDict[kmer][file] = 1

        # converting dictionary to dataframe
        dataset1 = pd.DataFrame.from_dict(kmersDict, orient='index')
        dataset1 = dataset1.fillna(0)
        dataset1 = dataset1.transpose()
        dataset1 = dataset1.reindex((dataset1.columns), axis=1)
        print(dataset1)
        ###########################Preparing inputs for Random Forest model################################################################

        sero = []
        for item in list(dataset1.index):
            sero.append(sra2sero[item])
        # classes =>
        Y = sero

        # Encoding target labels with integer value between 0 and n_classes-1
        labelencoder_y = LabelEncoder()
        Y_model = labelencoder_y.fit(Y)
        # Saving the Label Encoder for later Use
        labelencoder_file = open(odir+"/label_encoder", "wb")
        pickle.dump(Y_model, labelencoder_file)
        labelencoder_file.close()

        # Transforming the actual class to encoded values
        Y_model = Y_model.transform(Y)

        X_model = dataset1.iloc[:, :]

        clfs = RandomForestClassifier(n_estimators=500, random_state=1)
        clfs.fit(X_model, Y_model)

        features = list(X_model.columns)
        feats = {}  # a dict to hold feature_name: feature_importance

        for fea, importance in zip(features, clfs.feature_importances_):
            feats[fea] = importance  # add the name/value pair

        # Calculating Gini's importance for all the features
        importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
        # Sorting features according to their decreasing importance
        importances.sort_values(by='Gini-importance', ascending=False, inplace=True)


        selected_fea = importances.index[0:int(k)]
        X_fea = X_model.loc[:, selected_fea]
        # print(X_fea)

        features = list(X_fea.columns)
        with open(odir+"/selected_features.txt" , "w") as f1:
            for i in features:
                f1.write(str(i) + "\t")

        clfs2 = RandomForestClassifier(n_estimators=500)
        clfs2.fit(X_fea, Y_model)
        pickle.dump(clfs2, open(odir+"/final_model" , "wb"))


if __name__ == '__main__':
    main()


