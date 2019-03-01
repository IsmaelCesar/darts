#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 8 21:10:04 2017
@author: Juan Carlos Rodriguez-Gamboa
e-mail: ieju4n@gmail.com
Use this script for loading the dataset QWines (Three classes of wines)
"""

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 0.
loading libraries and defining some variables
"""
import os
import sys
sys.path.append("/data")
sys.path.append("../")
#import torch
#torch.multiprocessing.set_start_method('spawn', force=True)
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import logging
import tensorflow as tf
logging.getLogger("tensorflow").setLevel(logging.WARNING)
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import pickle
import time
import keras
import sklearn
from sklearn import preprocessing
import concurrent.futures
import logging
import tensorflow as tf
import csv
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from darts_for_wine.experiment_darts_wine import run_experiment_darts_wine as run_experiment
from darts_for_wine.experiment_darts_wine import args
from darts_for_wine.experiment_darts_wine import logging
from utils import PerclassAccuracyMeter
#import autokeras as ak

np.random.seed(1)
tf.set_random_seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

global tic
global actualDir,dataset,labels,names,labels_
global start_value,step,end_value
global ini_value,file_name,first_column,samp,numfiles
global tr_labels, tr_names, te_dataset, te_labels, te_names 
global ttvar,ngr,ncl,train_set,test_set
global resetv, ldataset 
pic_ = 0  #set on in 1 if you want to save the figures


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 4.
Auxiliary functions
"""

"""
4.1.
In this function is defined the initial conditions
"""
def resetv():
    global actualDir,dataset,labels,names,labels_
    global start_value,step,end_value,repetions
    global ini_value,file_name,first_column,samp
    file_name = os.path.basename(__file__)
    path_name = os.path.realpath(__file__)
    actualDir = path_name[:-len(file_name)]
    os.chdir(actualDir)
    dataset = []
    labels = []
    labels_ = []
    names = []
    #with the following configuration the process must train two models,
    #one model for each window
    #Window one: 160 - 1730 
    #Window two: 160 - 3300 
    samp=1  #1 for no sampling (use sampling to reduce the number of samples)
    first_column=2 #Columns 0 and 1 correspond to humidity and temperature. 
    ini_value = int(160/samp) #ignoring the baseline (first 160 samples) 
    start_value = int(1730/samp) #Defines the first window 
    step = int(1570/samp) #Window size 
    end_value = int(3300/samp) + 1 #last value  
    repetions = 1  

"""
4.2.
This function is for resampling (f>1)
"""
def resampling(array):
    global samp
    narray = np.empty((1,array.shape[1]))
    st_po = 0
    fi_po = samp    
    for k in range(int(array.shape[0]/samp)):    
        narray_ = array[st_po:fi_po,:].mean(axis=0)
        narray = np.append(narray, [narray_], axis=0)
        st_po+=samp
        fi_po+=samp
       
    narray =narray[1:,:]
    return narray

"""
4.3.
This function loads the files
"""    
def load_file(filename_):
    a_ = np.loadtxt(filename_, comments='#', delimiter='\t',dtype = np.float64)    
    if samp==1:
        f_=a_
    else:
        f_=resampling(a_)  #(10 new sampling)
    return f_
   
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 3.
The secondary function to load the dataset
"""
#load dataset
def ldataset(folder,lab,k_,pic,opt): 
    global actualDir,dataset,labels,names,end_value,file_name,labels_
    os.chdir(folder)
    with concurrent.futures.ProcessPoolExecutor() as executor:    
        filenames=os.listdir(os.getcwd()) 
        for filename, a in zip(filenames,executor.map(load_file,filenames)):
        ##save figure 
            if pic==1:
                os.chdir(actualDir)
                os.chdir('files/figures')           
                fg = plt.figure()
                plt.plot(a[:,first_column:])
                fg.savefig(filename[:-3]+'png', bbox_inches='tight',dpi=100)
                plt.clf() # Clear figure
                os.chdir(actualDir)    
                os.chdir(folder)
            ##save figure  
            dataset.append(a[0:end_value,first_column:])
            labels.append(lab)
            labels_.append(k_)
            names.append(filename)
            #del a
    os.chdir(actualDir)
    # Saving the objects:
    with open('QWines-Csystem' + opt + '.pkl', 'wb') as f:  
        pickle.dump([dataset,labels,labels_,names], f)
    print('loaded ' + folder)


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 5.
The main function to train the model
"""
def train_process(idx):
    global start_value,end_value,step
    global labels,tic,file_name,repetions,labels_
    global ini_value,last_column,numfiles
    global tr_labels, tr_names, te_dataset, te_labels, te_names 
    global ttvar,ngr,ncl,sizeT,train_set,test_set,flat_train_data,train_label,test_label,train_data,test_data
       
    train_results = {}
    test_results = {}
    etime = {}
    tic = time.time()



    for final_measurement in range(start_value, end_value+1, step):

        perclass_metter = PerclassAccuracyMeter(ngr)
        arg_lr = args.learning_rate
        arg_scheduler = None

        train_results[str(final_measurement)] = []
        test_results[str(final_measurement)] = []
        etime[str(final_measurement)] = []              
        
        indx=0
       
        #In this section is to perform the LOO
        model = None
        perclass_metter.first_iteration = True

        for i in range(ncl):
            test_set=[]
            train_set=[]
            tr_labels=[]
            te_labels=[]
            j=1
            h=1
            indx=indx+ttvar[ngr+2+i]-1
            for k in range(sizeT):
                if labels_[indx]==labels_[k]:
                    test_set.append(dataset[k])
                    te_labels.append(labels[k])
                    j+=1
                else:
                    train_set.append(dataset[k])
                    tr_labels.append(labels[k])
                    h+=1

            tr_labels = np.array(tr_labels)
            te_labels = np.array(te_labels)
            test_set = np.array(test_set)
            train_set = np.array(train_set)
            #Finish the LOO
            #repetitions = 10  # repetitions
            #for k in range(repetions):

            #Data shuffle
            train_data, train_label = sklearn.utils.shuffle(train_set[:,ini_value:final_measurement,:], tr_labels)
            test_data, test_label = sklearn.utils.shuffle(test_set[:,ini_value:final_measurement,:], te_labels)

    #                #preprocess
            flat_train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * last_column)
            flat_test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * last_column)
            scaler = preprocessing.StandardScaler().fit(flat_train_data)
            flat_train_data = scaler.transform(flat_train_data)
            flat_test_data = scaler.transform(flat_test_data)

            train_data = flat_train_data.reshape(train_data.shape[0], train_data.shape[1],train_data.shape[2], 1)
            test_data = flat_test_data.reshape(test_data.shape[0], train_data.shape[1],train_data.shape[2], 1)
            #input_shape = (train_data.shape[1],train_data.shape[2],1)

            # convert class vectors to binary class matrices
            cat_train_label = keras.utils.to_categorical(train_label,ngr)
            cat_test_label = keras.utils.to_categorical(test_label,ngr)
            num_classes=cat_train_label.shape[1]

            ##Put here the Convolutive CNN
            results_list, model,arg_scheduler= run_experiment(train_data, train_label, test_data,test_label,
                                                              perclass_metter,num_classes, model, final_measurement,
                                                             arg_lr,arg_scheduler)
            train_results[str(final_measurement)]+=np.array(results_list)[1:, 0].astype(float).tolist()
            test_results[str(final_measurement)]+=np.array(results_list)[1:, ngr*2+1].astype(float).tolist()
            perclass_metter.first_iteration = False
               
        etime_ = time.time() - tic
        etime[str(final_measurement)].append(etime_)
        logging.info("execution time: "+str(etime_))
        logging.info("Displyaing partial results:")
        #Display test results
        for dict_value in test_results.keys():
            logging.info('test2:')
            mean_acc_test = np.mean(test_results[dict_value])
            logging.info(dict_value+": "+str(mean_acc_test))
#        for dict_value in train_results.keys():
#            print('train2:')
#            mean_acc_train = np.mean(train_results[dict_value])
#            print(dict_value, mean_acc_train)
                             
# Saving the objects:
    with open('out_' + file_name[:-3] + idx +'.pkl', 'wb') as f:  
        pickle.dump([train_results,test_results,etime], f)
   
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 2.
The primary function to load the dataset.
"""
def calload(sys,p,opt,load):
    global dataset,labels,names,last_column,numfiles,labels_
    resetv()  
    if load:   
        k=0
        for i in range(len(sys)):
            for j in range(sys[i]):
                fold_ = 'wines/class' + str(i+1) + '/wine' + str(j+1)  + '/'  
                ldataset(fold_,i,k,p,opt)
                k+=1
    else:
        with open('../../data/wines/'+'QWines-Csystem' + opt + '.pkl', 'rb') as f_s:
            dataset,labels,labels_,names = pickle.load(f_s)
    dataset = np.array(dataset)
    labels = np.array(labels)
    labels_ = np.array(labels_)
    if opt=='TR':
        dim_data = dataset.shape
        print(dim_data)
        last_column = int(dim_data[2])
        numfiles = len(labels)
        print(str(numfiles)+' files loaded from Rwine')
    else:
        print(str(len(labels))+' files loaded from Rwine')
    return dataset,labels,labels_,names



"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 1.
The script begins here. 
"""
ttvar = [3,	43,	51,	141, 22, 9,	11,	13,	10,	10,	10,	10,	10,	11,	11,	11,	10,	11,	11,	10,	11,	12,	11,	10,	11,	11,	11]  #Wines thersholds
ngr=ttvar[0]
ncl=ttvar[ngr+1]
calload([4,5,13],pic_,'TR',0) #QWines-Csystem [4,5,13]
sizeT=len(dataset)
train_process('LOO')

