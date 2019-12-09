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
import sys
sys.path.append("/data")
sys.path.append("../")
import time

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
import pandas as pd

# Importing sources from the experiment module
import utils
from utils import PerclassAccuracyMeter
from darts_for_wine.experiment_darts_wine import args
from darts_for_wine.experiment_darts_wine import logging
from darts_for_wine.experiment_darts_wine import run_experiment_darts_wine as run_experiment
import time as time_formatter


   
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
global tic
global actualDir,dataset,labels,names,train_results
global test_results,start_value,step,end_value,repetitions
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
    global actualDir, dataset, labels, names, train_results, etime
    global valid_results, start_value, step, end_value, repetitions
    global ini_value, file_name, first_column, samp
    file_name = os.path.basename(__file__)
    path_name = os.path.realpath(__file__)
    actualDir = path_name[:-len(file_name)]
    os.chdir(actualDir)
    dataset = []
    labels = []
    names = []
    samp=1 #use sampling to reduce the number of samples
    first_column=1
    #with the following configuration the process must train two models,
    #one model for each window
    #Window one:  30 - 165
    #Window two:  30 - 299
    #Next test ini_value 30,start_value 50,step 20
    ini_value = int(30/samp) 
    start_value = int(50/samp) #old_start_value 165
    step = int(20/samp)        #old_step 135
    end_value = int(299/samp) 
    repetitions = args.epochs  #Set up the epochs
    train_results = {}
    valid_results = {}
    etime = {}

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
    #a_=np.loadtxt(filename_)
    a__ = pd.read_csv(filename_, sep='\t') 
    a_ = a__.iloc[:,first_column:].values
    if samp==1:
        f_=a_
    else:
        f_=resampling(a_)  #(new sampling)
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
                os.chdir('figures')           
                fg = plt.figure()
                plt.plot(a)
                print(filename[:-3] +'png')
                fg.savefig(filename[:-3] +'png', bbox_inches='tight',dpi=100)
                plt.clf() # Clear figure
                os.chdir(actualDir)    
                os.chdir(folder)
            ##save figure  
            dataset.append(a)
            labels.append(i)
            names.append(filename)
            del a
    os.chdir(actualDir)
    numfiles = len(labels)
    # Saving the objects:
    with open('preloaded_dataset.pkl', 'wb') as f:  
        pickle.dump([dataset, labels, names], f)
    print('loaded' + folder)

def train_model(final_measurement,k_):
    global start_value, end_value, step, valid_results, train_results
    global repetitions, labels, tic, idx_, tmp_test_acc
    global ini_value, file_name, last_column, numfiles
    # Making data necessary for training global variables
    global model, scheduler, lr, perclass_meter, classes_number, partial_results
    
    #split train and test data
    train_data, test_data, train_label, test_label = train_test_split(dataset[:, ini_value:final_measurement, :],
                                                                      labels, test_size=0.2)
    
    #preprocess
    flat_train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * last_column)
    flat_test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * last_column)
    scaler = preprocessing.StandardScaler().fit(flat_train_data)
    flat_train_data = scaler.transform(flat_train_data)
    flat_test_data = scaler.transform(flat_test_data)
    #Next two lines have been added by Ismael
    #Putting arrays back into their original shape
    stdd_train_data = flat_train_data.reshape(train_data.shape[0], train_data.shape[1], last_column)
    stdd_test_data = flat_test_data.reshape(test_data.shape[0], test_data.shape[1], last_column)
    #cat_train_label = to_categorical(train_label)
    #cat_test_label = to_categorical(test_label)

    
    ## ********** Put here the Convolutive CNN  **********
    h, model, scheduler =run_experiment(stdd_train_data,
                                        train_label,
                                        stdd_test_data,
                                        test_label,
                                        perclass_meter,
                                        classes_number,
                                        model,
                                        final_measurement,
                                        lr,
                                        scheduler)
    h1 = []
    for el in h[1:]:
        h1.append(el[0])
    train_results[str(final_measurement)] = np.array(h1).astype(float).tolist()
    h1 = []
    for el in h[1:]:
        h1.append(el[0])
    test_results[str(final_measurement)] = np.array(h1).astype(float).tolist()
    return 0

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 5.
The main function to train the model
"""
def train_process(idx):
    global start_value, end_value, step, valid_results, train_results, etime
    global repetitions, labels, tic, idx_, tmp_test_acc, file_name
    # Making data necessary for training global variables
    global model, scheduler, lr, perclass_meter, classes_number
    idx_=idx
    tic = time()

    partial_results = {}
    classes_number = 3
    lr = args.learning_rate

    for final_measurement in range(start_value, end_value+1, step):
        valid_results[str(final_measurement)] = []
        train_results[str(final_measurement)] = []
        etime[str(final_measurement)] = 0
        model = None
        scheduler = None
        partial_results[str(final_measurement)] = 0

        etic = time.time()
        perclass_meter = PerclassAccuracyMeter(classes_number)
        perclass_meter.first_iteration = True
        # logging.info("\n\t WINDOW + %s\n", final_measurement)

        tmp_test_acc = 0
        # for k in range(repetitions):
        train_model(final_measurement, 0)
        etime[str(final_measurement)] += time.time() - etic

  
    # etime =
    logging.info("execution time: "+str(time() - tic))

    logging.info("Partial Outcomes")
    for dict_value in valid_results.keys():
        logging.info('test:')
        mean_acc_test = np.mean(valid_results[dict_value])
        logging.info("Window "+str(dict_value) + " mean acc:" + str(mean_acc_test))
    for dict_value in train_results.keys():
        logging.info('train:')
        mean_acc_train = np.mean(train_results[dict_value])
        logging.info("Window "+str(dict_value) + " mean acc:" + str(mean_acc_train))

    with open('outcomes_'+file_name[:-3] +'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([train_results, valid_results, etime], f)
        f.close()

    

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 2.1
The primary function to load the dataset.
"""    
def call_ldataset(fold,clas,pic_):
    resetv()
    for i in range(len(clas)):
        print(fold+clas[i])
        ldataset(fold+clas[i],i,pic_) 

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 2.1
This function calls the script that loads the dataset 
and starts the training process
"""
def run_tr(fl_):
    global dataset, labels, names, last_column, first_column, numfiles
    resetv()
    if not names:
        with open(fl_ + '.pkl', 'rb') as f_s: 
            dataset, labels, names = pickle.load(f_s)
    dataset = np.array(dataset)
    dim_data = dataset.shape
    last_column = int(dim_data[2])
    numfiles = len(labels)
    print(str(numfiles)+'files loaded from ' + fl_)
    train_process(fl_) 


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 2.
This function calls the script that loads the dataset 
and starts the training process
""" 
def lauch(clas_,p):
    fold_='dataset/'
    print('loading ' + fold_) 
    #The next line could be omitted if the dataset is preloaded
    #call_ldataset(fold_,clas_,p) #Execute this line to load the dataset
    run_tr('../../data/coffee_dataset/preloaded_dataset')
    
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 1.
The script begins here. 
"""
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
#args.save ="EXP_DARTS_COFFEE"
args.save = '{}-{}-Coffee_DataSet_WithPrecisionRecallF1Score'.format(args.save, time_formatter.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

clas_=['AQ-Coffee','HQ-Coffee','LQ-Coffee']  #Classes
lauch(clas_, pic_) #loading the dataset
