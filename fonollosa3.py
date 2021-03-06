# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:29:08 2018

@author: usuario
"""

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 0.
loading libraries and defining some variables
"""
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import numpy as np
import pickle
import matplotlib.pyplot as plt
import concurrent.futures
#import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn import preprocessing
from keras import models
from keras import layers
from keras import regularizers
from keras import backend as K
from time import time
from keras.models import model_from_json
import csv
   
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
global tic
global actualDir,dataset,labels,names,train_results
global test_results,start_value,step,end_value,repetions
global ini_value,file_name,first_column,samp,numfiles
    
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
    global actualDir,dataset,labels,names,train_results
    global test_results,start_value,step,end_value,repetions
    global ini_value,file_name,first_column,samp
    file_name = os.path.basename(__file__)
    path_name = os.path.realpath(__file__)
    actualDir = path_name[:-len(file_name)]
    os.chdir(actualDir)
    dataset = []
    labels = []
    names = []
    samp=5 #use sampling to reduce the number of samples
    first_column=1
    #with the following configuration the process must train two models,
    #one model for each window
    #Window one:  5000 - 12000 
    #Window two:  5000 - 19000 
    ini_value = int(5000/samp) 
    start_value = int(12000/samp) #int(5500/samp) 
    step = int(7000/samp) 
    end_value = int(19000/samp) + 1  #19000 -> 19289 samples size of B5_GMe_F050_R1.txt file
    repetions = 5  #Set up the epochs
    train_results = {}
    test_results = {}

"""
4.2.
This function is for resampling (f>1)
"""
def resampling(array):
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
    a_=np.loadtxt(filename_)
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
def ldataset(folder,i,pic): 
    global actualDir,dataset,labels,names,end_value,file_name,numfiles
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
            labels.append(i)
            names.append(filename)
            del a
    os.chdir(actualDir)
    numfiles = len(labels)
    # Saving the objects:
    with open(folder[-5:-3] + '-system'+ '.pkl', 'wb') as f:  
        pickle.dump([dataset,labels,names], f)
    print('loaded' + folder)

def train_model(final_measurement,k_):
    global start_value,end_value,step,test_results,train_results
    global repetions,labels,tic,idx_,tmp_test_acc
    global ini_value,file_name,last_column,numfiles
    #split train and test data
    train_data, test_data, train_label, test_label = train_test_split(dataset[:,ini_value:final_measurement,:], labels, test_size = 0.5)
     
    #preprocess
    flat_train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * last_column)
    flat_test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * last_column)
    scaler = preprocessing.StandardScaler().fit(flat_train_data)
    flat_train_data = scaler.transform(flat_train_data)
    flat_test_data = scaler.transform(flat_test_data)
     
    cat_train_label = to_categorical(train_label)
    cat_test_label = to_categorical(test_label)
    
    ## ********** Put here the Convolutive CNN  **********
    
    
    # #creating the model
    # K.clear_session()
    # model = models.Sequential()
    # model.add(layers.Dense(100, activation='relu', input_shape=(flat_train_data.shape[1],)))
    # model.add(layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    # model.add(layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    # model.add(layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    # model.add(layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    # model.add(layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    # model.add(layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.02))) 
    # model.add(layers.Dense(4, activation='softmax'))
    # #model.add(layers.Dense(4, activation='linear'))
    # model.compile(optimizer='rmsprop',
    #                loss='categorical_crossentropy',
    #                metrics=['accuracy'])
    
    # history = model.fit(flat_train_data, cat_train_label, epochs = 200, batch_size = round(numfiles*0.2)  #10  
    #                      , verbose=False)
    # #testing the trained model
    # test_loss, test_acc = model.evaluate(flat_test_data, cat_test_label, verbose=False)
    # train_loss, train_acc = model.evaluate(flat_train_data, cat_train_label, verbose=False)
    # train_results[str(final_measurement)].append(train_acc)
    # test_results[str(final_measurement)].append(test_acc)
    # np.save('test_' + file_name[:-3] + idx_, test_results)
    # np.save('train_' + file_name[:-3] + idx_, train_results)
   
    #Saving the model
    # if test_acc>tmp_test_acc:
    #     # serialize model to JSON
    #     model_json = model.to_json()
    #     with open('model_'+ file_name[:-3] + idx_ + '.json', 'w') as json_file:
    #         json_file.write(model_json)
    #     # serialize weights to HDF5
    #     model.save_weights('model_' + file_name[:-3] + idx_ + '.h5')
    #     tmp_test_acc=test_acc
    
    ##Loading the saved model
     
    # load json and create model
#    json_file = open('modelB4-8000.json', 'r')
#    loaded_model_json = json_file.read()
#    json_file.close()
#    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
#    loaded_model.load_weights("model.h5")
#    print("Loaded model from disk")
     
    # evaluate loaded model on test data
#    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#    score = loaded_model.evaluate(flat_test_data, cat_test_label, verbose=False)
#    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))      
#    
         
    return history

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 5.
The main function to train the model
"""
def train_process(idx):
    global start_value,end_value,step,test_results,train_results
    global repetions,labels,tic,idx_,tmp_test_acc,file_name
    idx_=idx
    tic = time()
    
    for final_measurement in range(start_value, end_value+1, step):
        test_results[str(final_measurement)] = []
        train_results[str(final_measurement)] = []
      
        tmp_test_acc=0      
        for k in range(repetions):
            train_model(final_measurement,k)
            #early stopping
            if tmp_test_acc==1:
                break                
           
  
    etime = time() - tic
    print("execution time: "+str(etime))
    
    ##Printing partial outcomes  
    #for dict_value in test_results.keys():
    #    print('test:')
    #    mean_acc_test = np.mean(test_results[dict_value])
    #    print(dict_value, mean_acc_test)
    #for dict_value in train_results.keys():
    #    print('train:')
    #    mean_acc_train = np.mean(train_results[dict_value])
    #    print(dict_value, mean_acc_train)
    
    ## Saving the outcomes:
    # fcsv=file_name[:-3]+'.csv'      
    # with open(fcsv, 'w') as csvfile:
    #     spamwriter = csv.writer(csvfile, delimiter='\t',
    #                             quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    #     spamwriter.writerow(['TEST', 'Time' , 'Size'])
    #     spamwriter.writerow([ actualDir, etime , len(test_results.keys())])
    #     spamwriter.writerow(['final measurement'  , 'mean', 'std', ])
    
    #     for dict_value in test_results.keys():
    #         mean_acc_test = np.mean(test_results[dict_value])
    #         std_acc_test = np.std(test_results[dict_value])
    #         spamwriter.writerow([dict_value,  mean_acc_test,  std_acc_test])
            
    #     spamwriter.writerow(['TRAIN', 'Time' , 'Size'])
    #     spamwriter.writerow([ actualDir, etime , len(train_results.keys())])
    #     spamwriter.writerow(['final measurement'  , 'mean', 'std', ])
            
    
    #     for dict_value in train_results.keys():
    #         mean_acc_train = np.mean(train_results[dict_value])
    #         std_acc_train = np.std(train_results[dict_value])
    #         spamwriter.writerow([dict_value,  mean_acc_train,  std_acc_train])      
        
    ## Saving the objects:
    # with open('outcomes_'+ file_name[:-3] + idx_ +'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([train_results,test_results,etime], f)

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 2.1
The primary function to load the dataset.
"""    
def call_ldataset(fold,clas,pic_):
    resetv()
    for i in range(len(clas)):
        ldataset(fold+clas[i],i,pic_) 

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 2.1
This function calls the script that loads the dataset 
and starts the training process
"""
def run_tr(fl_):
    global dataset,labels,names,last_column,first_column,numfiles
    resetv()
    if not names:
        with open(fl_ + '-system'+ '.pkl', 'rb') as f_s: 
            dataset,labels,names = pickle.load(f_s)
    dataset = np.array(dataset)
    dim_data = dataset.shape
    last_column = int(dim_data[2])
    numfiles = len(labels)
    print(str(numfiles)+'files loaded from ' + fl_)
    train_process(fl_) 

#with open('train&test_resultsB1.pkl', 'rb') as f_s: 
#    train_results,test_results,mean_acc_train,mean_acc_test,etime = pickle.load(f_s)

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 2.
This function calls the script that loads the dataset 
and starts the training process
""" 
def lauch(sys,clas_,p):
    syst_=['B1','B2','B3','B4','B5']
    fold_='files/' + str(syst_[sys]) + '/'   
    print('loading ' + fold_)
    #call_ldataset(fold_,clas_,p)
    run_tr(syst_[sys])
    
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 1.
The script begins here. 
"""

clas_=['CO','Ea','Ey','Me']  #Classes
lauch(4,clas_,pic_) #Execute this line to load B5 dataset  
#lauch(3,clas_,pic_) #Execute this line to load B4 dataset  
#lauch(2,clas_,pic_) #Execute this line to load B3 dataset  
#lauch(1,clas_,pic_) #Execute this line to load B2 dataset  
#lauch(0,clas_,pic_) #Execute this line to load B1 dataset  
