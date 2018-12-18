#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 5 21:10:04 2018
@author: Juan Carlos Rodriguez-Gamboa
e-mail: ieju4n@gmail.com
"""
import os
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
#import autokeras as ak

#from keras import backend as K

np.random.seed(1)
tf.set_random_seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

global tic
global actualDir,dataset,labels,names,labels_
global start_value,step,end_value
global ini_value,file_name,first_column,samp,numfiles
global tr_labels, tr_names, te_dataset, te_labels, te_names 
global ttvar,ngr,ncl,train_set,test_set
global running,REF
global resetv, ldataset 

pic_ = 0  #set on in 1 if you want to save the figures
 
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
    samp=1
    first_column=2
    ini_value = int(160/samp) 
    start_value = int(1570/samp) 
    step = int(1570/samp) 
    end_value = int(3300/samp) + 1   
    repetions = 1

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
    
def load_file(filename_):
    a_ = np.loadtxt(filename_, comments='#', delimiter='\t',dtype = np.float64)    
    if samp==1:
        f_=a_
    else:
        f_=resampling(a_)  #(10 new sampling)
    return f_
   
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


def train_process(idx):
    global start_value,end_value,step
    global labels,tic,file_name,repetions,labels_
    global ini_value,last_column,numfiles
    global tr_labels, tr_names, te_dataset, te_labels, te_names 
    global ttvar,ngr,ncl,sizeT,train_set,test_set,REF,flat_train_data,train_label,test_label,train_data,test_data
       
    train_results = {}
    test_results = {}
    etime = {}
    tic = time.time()
    
    for final_measurement in range(start_value, end_value+1, step):
        
        train_results[str(final_measurement)] = []
        test_results[str(final_measurement)] = []
        etime[str(final_measurement)] = []              
        
        indx=0

        #Leave one out -LOO-      
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

       
            for k in range(repetions):
                                             
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
                input_shape = (train_data.shape[1],train_data.shape[2],1)
              
                # convert class vectors to binary class matrices
                cat_train_label = keras.utils.to_categorical(train_label,num_classes=ngr)
                cat_test_label = keras.utils.to_categorical(test_label,num_classes=ngr)
                num_classes=cat_train_label.shape[1]
                
#                #create model
#                keras.backend.clear_session()
#                model = keras.models.Sequential()
#                ##Convolutive
        
               
                
                
        etime_ = time.time() - tic
        etime[str(final_measurement)].append(etime_)
        print("execution time: "+str(etime_))
     

                             
# Saving the objects:
#    with open('out_' + file_name[:-3] + str(REF) + idx +'.pkl', 'wb') as f:  
#        pickle.dump([train_results,test_results,etime], f)
   

def calload(sys,p,opt,load):
    global dataset,labels,names,last_column,numfiles,labels_
    file_path = "../../data/wines/" #file path to the data folder of the DARTS implementation
    resetv()  
    if load:   
        k=0
        for i in range(len(sys)):
            for j in range(sys[i]):
                fold_ = 'wines/class' + str(i+1) + '/wine' + str(j+1)  + '/'  
                ldataset(fold_,i,k,p,opt)
                k+=1
    else:
        with open(file_path + opt + '.pkl', 'rb') as f_s:
            dataset,labels,labels_,names = pickle.load(f_s)
    dataset = np.array(dataset)
    labels = np.array(labels)
    labels_ = np.array(labels_)
    if opt=='QWines-CsystemTR':
        dim_data = dataset.shape
        print(dim_data)
        last_column = int(dim_data[2])
        numfiles = len(labels)
        print(str(numfiles)+' files loaded from Rwine')
    else:
        print(str(len(labels))+' files loaded from Rwine')
    return dataset,labels,labels_,names


#QWines-Csystem
ttvar = [3,	43,	51,	141, 22,	9,	11,	13,	10,	10,	10,	10,	10,	11,	11,	11,	10,	11,	11,	10,	11,	12,	11,	10,	11,	11,	11]  #Wines thersholds
#QWinesEa-Csystem
#ttvar = [4,	43,	65,	51,	141,	28,	9,	11,	13,	10,	10,	11,	12,	11,	10,	11,	10,	10,	10,	10,	11,	11,	11,	10,	11,	11,	10,	11,	12,	11,	10,	11,	11,	11] ##Wines thersholds + Ethanol
ngr=ttvar[0]
ncl=ttvar[ngr+1]
calload([4,5,13],pic_,'QWines-CsystemTR',0) #QWines-Csystem [4,5,13] QWinesEa-Csystem [4,6,5,13]
sizeT=len(dataset)
train_process('LOO') 





