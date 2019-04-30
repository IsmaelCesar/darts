# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:29:08 2018

@author: usuario
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
sys.path.append("../")
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn import preprocessing
from keras import models
from keras import layers
from keras import regularizers
from keras import backend as K
from time import time
import csv
from keras.models import model_from_json
from sklearn.externals import joblib
import pandas as pd 
from sklearn.decomposition import PCA

#Added By ismael
import utils
from darts_for_wine.experiment_darts_wine import logging
from darts_for_wine.experiment_darts_wine import args
from darts_for_wine.experiment_darts_wine import run_experiment_darts_wine as run_experiment
from darts_for_wine.experiment_darts_wine import infer
from darts_for_wine.winedataset import WinesDataset
from darts_for_wine.test_cases import testing_csv_list
import torch
import torch.nn as nn
import torch.utils.data as torchdata
import time as time_formatter

   
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
global tic,pic_
global actualDir,dataset,labels,names,train_results,test_results
global valid_results,start_value,step,end_value,repetions
global ini_value,file_name,first_column,samp,numfiles

pic_ = 0  #set on in 1 if you want to save the figures
 
def resetv():
    global actualDir,dataset,labels,names,train_results,test_results
    global valid_results,start_value,step,end_value,repetions
    global ini_value,file_name,first_column,samp
    global datasetT,labelsT,namesT
    
    file_name = os.path.basename(__file__)
    path_name = os.path.realpath(__file__)
    actualDir = path_name[:-len(file_name)]
    os.chdir(actualDir)
    dataset = []
    datasetT = []
    labels = []
    labelsT = []
    names = []
    namesT = []
    samp=1 #1Hz
    first_column=0
    #with the following configuration the process must train two models,
    #one model for each window
    #Window one:  20 - 80 
    #Window two:  20 - 140
    #Window three:  20 - 200
    #Window four:  20 - 260
    if (samp==0):
        samp_=1
    else:
        samp_=samp
    ini_value = int(20*samp_) 
    start_value = int(80*samp_) #int(5500/samp) old_start_value 60*samp_
    step = int(60*samp_) #old_step 40*samp_
    end_value = int(260*samp_)  #samples size old end_value 260*samp_
    repetions = 1
    train_results = {}
    valid_results = {}
    test_results = {}

def resampling(array):
    global samp
    narray = np.empty((1,array.shape[1]-1))
    narray_bl = np.empty((1,array.shape[1]-1))
    st_po = 0
    rs=np.where(array[:,0]<=1000/samp)
    fi_po = max(rs[0])
    
    for k in range(260*samp):    
        narray_ = array[st_po:fi_po,1:].mean(axis=0)
        narray = np.append(narray, [narray_], axis=0)
        st_po = fi_po - 1 
        rs=np.where(array[:,0]<=((1000/samp)*(k+2)))
        if (rs[0].size>0):
            fi_po = max(rs[0])
        else:
            fi_po = fi_po +1
       
    narray =narray[1:,:]
    
    #removing baseline
    array_bl = narray[0:20*samp,:].mean(axis=0)
    narray_bl = narray/array_bl #relativa 
             
    return narray_bl


def load_file(filename_):
    #a_=np.loadtxt(filename_)
    if (bool(filename_.find('board_setPoint_500V')+1) and bool(filename_.find('fan_setPoint_060')+1) == True):
        sel_ =[ 0.,   12.,
           13., 14., 15., 16., 17., 18., 19., 21., 22., 23., 24., 25.,
           26., 27., 28., 30., 31., 32., 33., 34., 35., 36., 37., 
           39., 40., 41., 42., 43., 44., 45., 46., 49., 50., 51.,  #48.,
           52., 53., 54., 55., 57., 58., 59., 60., 61., 62., 63., 64.,
           66., 67., 68., 69., 70., 71., 72., 73., 75., 76., 77.,
           78., 79., 80., 81., 82., 84., 85., 86., 87., 88., 89., 90.,
           91.]
        a__ = pd.read_csv(filename_, sep='\t') 
        a_ = a__.iloc[:,sel_].values
        if samp==0:
            f_=a_
        else:
            f_=resampling(a_)  #(10 new sampling)
        return f_
   
   
#load dataset
def ldataset(sel,fl_,folder,i): 
    global actualDir,dataset,labels,names,end_value,file_name,numfiles,pic_
    global datasetT,labelsT,namesT,numfilesT
    os.chdir(folder)
    with concurrent.futures.ProcessPoolExecutor() as executor:    
        filenames=os.listdir(os.getcwd()) 
        for filename, a in zip(filenames,executor.map(load_file,filenames)):
            if (bool(filename.find('board_setPoint_500V')+1) and bool(filename.find('fan_setPoint_060')+1) == True):
                ##save figure 
                if pic_==1:
                    os.chdir(actualDir)
                    os.chdir('WTD_files/figures/' + fl_ )           
                    fg = plt.figure()
                    plt.plot(a[:,first_column:])
                    fg.savefig(filename[:-3]+'png', bbox_inches='tight',dpi=100)
                    plt.clf() # Clear figure
                    os.chdir(actualDir)    
                    os.chdir(folder)
                ##save figure  
                if (sel==0):
                    dataset.append(a[:,first_column:])
                    labels.append(i)
                    names.append(filename)
                else:
                    datasetT.append(a[:,first_column:])
                    labelsT.append(i)
                    namesT.append(filename)
                del a
    os.chdir(actualDir)
    if (sel==0):
        numfiles = len(labels)
        # Saving the objects:
        with open('preloaded_dataset-' + fl_ + '.pkl', 'wb') as f:  
            pickle.dump([dataset,labels,names], f)
    else:
        numfilesT = len(labelsT)
        # Saving the objects:
        with open('preloaded_dataset-' + fl_ + '.pkl', 'wb') as f:  
            pickle.dump([datasetT,labelsT,namesT], f)    
    print('loaded' + folder)

def train_model(final_measurement,k_,te_g):
    global start_value,end_value,step,valid_results,train_results,test_results
    global repetions,labels,tic,tr_g,tmp_valid_acc
    global ini_value,file_name,last_column,numfiles
    global datasetT,labelsT,namesT
    global dim_dataT,last_columnT
    # Added by Ismael
    global model, scheduler, lr, perclass_meter, classes_number
    
    #split train and validation data
    train_data, valid_data, train_label, valid_label = train_test_split(dataset[:,ini_value:final_measurement,:], labels, test_size = 0.3)
    #test data
    test_data = datasetT[:,ini_value:final_measurement,:]
    test_label = labelsT
     
    #preprocess
    flat_train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * last_column)
    flat_valid_data = valid_data.reshape(valid_data.shape[0], valid_data.shape[1] * last_column)
    flat_test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * last_columnT)
    scaler = preprocessing.StandardScaler().fit(flat_train_data)
    flat_train_data = scaler.transform(flat_train_data)
    scaler1 = preprocessing.StandardScaler().fit(flat_valid_data)
    flat_valid_data = scaler1.transform(flat_valid_data)
    scaler2 = preprocessing.StandardScaler().fit(flat_test_data)
    flat_test_data = scaler2.transform(flat_test_data)
      
    #cat_train_label = to_categorical(train_label)
    #cat_valid_label = to_categorical(valid_label)
    #cat_test_label = to_categorical(test_label)
    #Added By Ismael
    stdd_train_data = flat_train_data.reshape(train_data.shape[0],train_data.shape[1],last_column)
    stdd_valid_data = flat_valid_data.reshape(valid_data.shape[0], valid_data.shape[1], last_column)
    stdd_test_data = flat_test_data.reshape(test_data.shape[0], test_data.shape[1], last_columnT)
    ## ********** Put here the Convolutive CNN  **********
    h,model,scheduler = run_experiment(stdd_train_data,train_label,stdd_valid_data,valid_label,perclass_meter,
                                      classes_number,model,final_measurement,lr,scheduler)

    logging.info("\t\t\n\n USING TEST SET OF WINDOW"+str(final_measurement)+"\n\n")

    dset_obj = WinesDataset(stdd_test_data,test_label)
    test_queue = torch.utils.data.DataLoader(dset_obj, sampler=torchdata.sampler.RandomSampler(dset_obj),
                                                  pin_memory=True, num_workers=2)
    infer(test_queue,model,nn.CrossEntropyLoss(),classes_number)

    #h = testing_csv_list(perclass_meter,labels,args.save,final_measurement)


    train_results[str(final_measurement)] = np.array(h)[1:, 0].astype(float).tolist()
    test_results[str(final_measurement)] = np.array(h)[1:, classes_number * 2 + 1].astype(float).tolist()

    return 0


def train_process(te_g):
    global start_value,end_value,step,valid_results,train_results,test_results
    global repetions,labels,tic,tr_g,tmp_valid_acc
    #Added by Ismael
    global clas_,model, scheduler, lr, perclass_meter, classes_number

    tic = time()

    classes_number = len(clas_)

    for final_measurement in range(start_value, end_value+1, step):
        valid_results[str(final_measurement)] = []
        train_results[str(final_measurement)] = []
        test_results[str(final_measurement)] = []


        #Perclass Metter
        perclass_meter = utils.PerclassAccuracyMeter(classes_number,is_using_prf1=True)
        perclass_meter.first_iteration = True
        model = None
        scheduler = None
        lr = args.learning_rate

        tmp_valid_acc=0   
        #for k in range(repetions):
        train_model(final_measurement,k,te_g)
        #early stopping
        #if tmp_valid_acc==1:
        #    break
        
  
    etime = time() - tic
    logging.info("execution time: "+str(etime))

    #Printing partial outcomes
    for dict_value in valid_results.keys():
       logging.info('valid:')
       mean_acc = np.mean(valid_results[dict_value])
       logging.info("Window "+str(dict_value) + " mean acc:" + str(mean_acc))
       #mean_acc_valid = np.mean(valid_results[dict_value])
    for dict_value in train_results.keys():
       logging.info('train:')
       mean_acc_train = np.mean(train_results[dict_value])
       logging.info("Window "+str(dict_value)+"mean acc:"+str(mean_acc_train))

#    # Saving the outcomes:
#    fcsv= 'summary_'+ file_name[:-3] + tr_g + te_g + '.csv'      
#    with open(fcsv, 'w') as csvfile:
#        spamwriter = csv.writer(csvfile, delimiter='\t',
#                                 quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#        
#        spamwriter.writerow(['TEST', 'Time' , 'Size'])
#        spamwriter.writerow([ actualDir, etime , len(test_results.keys())])
#        spamwriter.writerow(['final measurement'  , 'mean', 'std', ])
#                
#        for dict_value in test_results.keys():
#            mean_acc_test = np.mean(test_results[dict_value])
#            std_acc_test = np.std(test_results[dict_value])
#            spamwriter.writerow([dict_value,  mean_acc_test,  std_acc_test])
#                
#        spamwriter.writerow(['VALID', 'Time' , 'Size'])
#        spamwriter.writerow([ actualDir, etime , len(valid_results.keys())])
#        spamwriter.writerow(['final measurement'  , 'mean', 'std', ])
#    
#        for dict_value in valid_results.keys():
#            mean_acc_valid = np.mean(valid_results[dict_value])
#            std_acc_valid = np.std(valid_results[dict_value])
#            spamwriter.writerow([dict_value,  mean_acc_valid,  std_acc_valid])
#            
#        spamwriter.writerow(['TRAIN', 'Time' , 'Size'])
#        spamwriter.writerow([ actualDir, etime , len(train_results.keys())])
#        spamwriter.writerow(['final measurement'  , 'mean', 'std', ])
#      
#        for dict_value in train_results.keys():
#            mean_acc_train = np.mean(train_results[dict_value])
#            std_acc_train = np.std(train_results[dict_value])
#            spamwriter.writerow([dict_value,  mean_acc_train,  std_acc_train])      
#         
#    # Saving the objects:
#    with open('outcomes_'+ file_name[:-3] + tr_g + te_g +'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#        pickle.dump([train_results,valid_results,test_results,etime], f)

    
def call_ldataset():
    global clas_, fold_
    global tr_g
    
    resetv()
    #training group
    if not names:
        for i in range(len(clas_)):
            if (clas_[i]!='Butanol_100'):
                ldataset(0,tr_g,fold_+clas_[i]+'/'+ tr_g +'/',i) 
            else:
                if (tr_g != 'L6'):
                    ldataset(0,tr_g,fold_+clas_[i]+'/'+ tr_g +'/',i) 
    
    #testing group
    for j in range(len(clas_)):
        if (clas_[j]!='Butanol_100'):
            ldataset(1,te_g,fold_+clas_[j]+'/'+ te_g +'/',j) 
        else:
            if (te_g != 'L6'):
                ldataset(1,te_g,fold_+clas_[j]+'/'+ te_g +'/',j) 


def run_tr():
    global dataset,labels,names,last_column,first_column,numfiles
    global datasetT,labelsT,namesT
    global dim_dataT,last_columnT,tr_g,numfilesT
    resetv()
    #nextline was added by ismael
    prefix_path = "../../data/windtunel/"
    if not names:
        with open(prefix_path+'preloaded_dataset-' + tr_g + '.pkl', 'rb') as f_s:
            dataset,labels,names = pickle.load(f_s)
    dataset = np.array(dataset)
    dim_data = dataset.shape
    last_column = int(dim_data[2])
    numfiles = len(labels)
    print(str(numfiles)+'files loaded from' + tr_g)
    
    if not namesT:
        with open(prefix_path+'preloaded_dataset-' + te_g + '.pkl', 'rb') as f_s1:
            datasetT,labelsT,namesT = pickle.load(f_s1)
    datasetT = np.array(datasetT)
    dim_dataT = dataset.shape
    last_columnT = int(dim_data[2])
    numfilesT = len(labelsT)
    print(str(numfiles)+'files loaded from' + te_g)
    
    train_process(te_g) 
    


global clas_, syst_, fold_, tr_g, te_g
clas_=['Acetaldehyde_500','Acetone_2500','Ammonia_10000','Benzene_200','Butanol_100','CO_1000','CO_4000','Ethylene_500','Methane_1000','Methanol_200','Toluene_200']
syst_=['L1','L2','L3','L4','L5','L6']
fold_='WTD_files/'
tr_g=syst_[3] #Training group 'L4'


#Testing groups 'L1','L2','L3','L4','L5','L6'
for k in range(0, 6, 1):
    print('loading ' + fold_)

    # Added by ismael
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                       format=log_format, datefmt='%m/%d %I:%M:%S %p')
    args.save = "EXP_DARTS"

    args.save = ('search-{}-{}-WindTunel_'+syst_[k]+"PrecisionRecallF1Score").format(args.save, time_formatter.strftime("%Y%m%d-%H%M%S"))

    utils.create_exp_dir(args.save)

    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    te_g=syst_[k]
    print(te_g)
    #call_ldataset() #Only excute for non preloaded dataset
    run_tr()




