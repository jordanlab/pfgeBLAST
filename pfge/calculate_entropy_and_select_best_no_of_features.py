#!/usr/bin/env python3

import math
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
    parser.add_argument("--impt_dir")
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
    if (args.impt_dir):
        impt_dir = args.impt_dir


#############################Getting PFGE Mapping and Serotype Mapping into dictionary###########################################

    Serotypes_count_dict = {}
    PFGEs_count_dict = {}
    with open(PFGEmapping, "r") as f1:
        next(f1)  # ignore header
        for line in f1.readlines():
            line = line.strip("\n").split("\t")
            # line[0] => SRA id
            # line[1] => PFGE pattern id
            PFGEs_count_dict[line[1]] = PFGEs_count_dict.get(line[1], 0) + 1
            Serotypes_count_dict[line[1][0:3]] = Serotypes_count_dict.get(line[1][0:3], 0) + 1

    entropy_dict = {}
    cnt = 0
    for s in Serotypes_count_dict.keys():
        entropy = 0
        denominator = Serotypes_count_dict[s]
        #print(denominator)
        for k in PFGEs_count_dict.keys():
            if k[0:3] == s:
                #print(k)
                # print(k[0:3])
                entropy = ((PFGEs_count_dict[k] / denominator) * (math.log(PFGEs_count_dict[k] / denominator, 10)))
                #print((PFGEs_count_dict[k] / denominator))
                #print(entropy)
                entropy_dict[k[0:3]] = entropy_dict.get(k[0:3], 0) - entropy
                # print(entropy_dict)
    print(entropy_dict)


    f=open(odir+"/selected_no_of_feats", "w")

    f.write("Sero"+"\t"+"number_of_fea"+"\t"+"Accuracy"+"\n")

    for k in entropy_dict.keys():
        if entropy_dict[k] < 1:
            with open(impt_dir + "/impt_vs_cv_accuracy_%s.txt" %k, "r") as fh:
                arr = []
                for line in fh:
                    # print(line)
                    line = line.rstrip().split("\t")
                    arr.append("{0:.3f}".format(float(line[2])))
            #print(arr)
            max_value = max(arr)  # maximum value
            max_keys = arr.index(max_value)  # getting all keys containing the `maximum`

            f.write(k+"\t"+ str(10+10*max_keys) +"\t" +str(max_value)+"\n")

        elif entropy_dict[k] >=  1:
            arr = []
            with open(impt_dir+"/impt_vs_cv_accuracy_%s.txt" %k, "r") as fh:
                for line in fh:
                    # print(line)
                    line = line.rstrip().split("\t")
                    arr.append(float(line[2]))

            arr = np.array(arr)
            #print(arr)
            indices = (arr.argsort()[-50:][::-1])

            #print(indices)

            range_dict = {}
            # indices = [6]
            for i in indices:
                # print(arr[i])
                if i - 15 < 0:
                    range_dict[i] = arr[0:i + 15]
                elif i + 15 > len(arr):
                    range_dict[i] = arr[i - 15:len(arr)]
                else:
                    range_dict[i] = arr[i - 15:i + 15]
                # print(range_dict)

            #print(range_dict)

            def select_best(range_dict, indices):
                return_val = -float("inf")
                K = None
                for k, v in range_dict.items():
                    MAX = float(max(v))
                    MIN = float(min(v))

                    # print(MAX, MIN, arr[k],v,k)

                    if ((MAX - float(arr[k])) / float(arr[k])) < 0.05:
                        # print("here")
                        if ((float(arr[k]) - MIN) / float(arr[k])) < 0.05:
                            if float(arr[k]) >= return_val:
                                return_val = float(arr[k])
                                K = k
                                # print(MIN, MAX, K, return_val)

                return 10+10*K, return_val

            fea, acc = select_best(range_dict, indices)
            f.write(k+"\t"+str(fea)+"\t"+str(acc)+"\n")

    f.close()




if __name__ == '__main__':
    main()





