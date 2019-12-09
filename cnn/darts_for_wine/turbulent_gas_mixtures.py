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
# import matplotlib.pyplot as plt
import os
import sys
import torch
sys.path.append("../")
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from sklearn import preprocessing
from keras import models
from keras import layers
from keras import regularizers
from keras import backend as K
import time
from keras.models import model_from_json
#import csv
import pandas as pd

#Added By ismael
import utils
from darts_for_wine.experiment_darts_wine import logging
from darts_for_wine.experiment_darts_wine import args
from darts_for_wine.experiment_darts_wine import run_experiment_darts_wine as run_experiment

   
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
global tic
global actualDir, dataset, labels, names, train_results
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
    global actualDir, dataset, labels, names, train_results
    global test_results, start_value, step, end_value, repetitions
    global ini_value, file_name, first_column, samp
    file_name = os.path.basename(__file__)
    path_name = os.path.realpath(__file__)
    actualDir = path_name[:-len(file_name)]
    os.chdir(actualDir)
    dataset = []
    labels = []
    names = []
    samp = 1 # use sampling to reduce the number of samples
    first_column = 3
    # with the following configuration the process must train two models,
    # one model for each window
    # Window one:  600 - 1785
    # Window two:  1785 - 2970
    ini_value = int(600/samp) # old value int(600/samp)
    start_value = int(837/samp) # old value int(1785/samp)
    step = int(236/samp) # old value int(1185/samp)
    end_value = int(2970/samp) # old value int(2970/samp)
    repetitions = 1  #Set up the epochs
    train_results = {}
    test_results = {}

"""
4.2.
This function is for resampling (f>1)
"""
def resampling(array):
    narray = np.empty((1, array.shape[1]))
    st_po = 0
    fi_po = samp    
    for k in range(int(array.shape[0]/samp)):    
        narray_ = array[st_po:fi_po, :].mean(axis=0)
        narray = np.append(narray, [narray_], axis=0)
        st_po += samp
        fi_po += samp
        
    narray = narray[1:, :]
    
    return narray

"""
4.3.
This function loads the files
"""  
def load_file(filename_):
    #a_=np.loadtxt(filename_)
    a__ = pd.read_csv(filename_) 
    a_ = a__.iloc[0:end_value, first_column:].values
    
    
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
                os.chdir('figures')           
                fg = plt.figure()
                plt.plot(a)
                fg.savefig(filename, bbox_inches='tight',dpi=100)
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
        pickle.dump([dataset,labels,names], f)
    print('loaded' + folder)

def train_model(final_measurement,k_):
    global start_value, end_value, step, test_results, train_results
    global repetitions, labels, tic, idx_, tmp_test_acc
    global ini_value, file_name, last_column, numfiles
    #Added By Ismael
    global num_classes, perclass_meter, model, scheduler, lr
    #split train and test data
    train_data, test_data, train_label, test_label = train_test_split(dataset[:, ini_value:final_measurement, :],
                                                                      labels, test_size=0.2) # 0.3
     
    #preprocess
    flat_train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * last_column)
    flat_test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * last_column)
    scaler = preprocessing.StandardScaler().fit(flat_train_data)
    flat_train_data = scaler.transform(flat_train_data)
    flat_test_data = scaler.transform(flat_test_data)

    stdd_train_data = flat_train_data.reshape(train_data.shape[0], train_data.shape[1], last_column)
    stdd_test_data = flat_test_data.reshape(test_data.shape[0], test_data.shape[1], last_column)
    
    ## ********** Put here the Convolutive CNN  **********
    h, model, scheduler = run_experiment(stdd_train_data, train_label, stdd_test_data, test_label, perclass_meter,
                                         num_classes, model, final_measurement, lr, scheduler)
    h1 = []
    for el in h[1: ]:
        h1.append(el[0])
    train_results[str(final_measurement)] = np.array(h1).astype(float).tolist()
    h1 = []
    for el in h[1:]:
        h1.append(el[0])
    test_results[str(final_measurement)] = np.array(h1).astype(float).tolist()

    # cat_train_label = to_categorical(train_label)
    #cat_test_label = to_categorical(test_label)
    stdd_test_data = stdd_test_data.reshape(stdd_test_data.shape[0], 1,
                                            stdd_test_data.shape[1], stdd_test_data.shape[2])

    preds = model(torch.FloatTensor(stdd_test_data).cuda()).cpu().detach().numpy()

    clsf_report = classification_report(test_label, np.argmax(preds, axis=1))

    logging.info("Cassification report table window "+str(final_measurement))
    logging.info("\n\n"+clsf_report+"\n\n")

    with open(args.save+'/'+"precision_recall_f1score_table_window"+str(final_measurement)+".txt", 'w+') as f:
        f.write(clsf_report)
        f.close()

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
         
    # return history
    return 0

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 5.
The main function to train the model
"""
def train_process(idx):
    global start_value, end_value, step, test_results, train_results
    global repetitions, labels, tic, idx_, tmp_test_acc, file_name
    #Added by Ismael
    global num_classes, perclass_meter, model, scheduler, lr
    idx_=idx
    outer_tic = time.time()

    etime = {}
    for final_measurement in range(start_value, end_value, step):# end_value+1
        test_results[str(final_measurement)] = []
        train_results[str(final_measurement)] = []
        etime[str(final_measurement)] = 0

        tic = time.time()
        # Perclass Metter
        perclass_meter = utils.PerclassAccuracyMeter(num_classes)
        perclass_meter.first_iteration = True
        model = None
        scheduler = None
        lr = args.learning_rate

        tmp_test_acc=0      
        # for k in range(repetitions):
        train_model(final_measurement, 0)
        etime[str(final_measurement)] += time.time() - tic
            #early stopping
        #    if tmp_test_acc==1:
        #        break
           
  
    # etime = time() - tic
    logging.info("total execution time: "+str(time.time() - outer_tic))
    
    ##Printing partial outcomes
    logging.info("Partial Outcomes")
    for dict_value in test_results.keys():
         logging.info('test:')
         mean_acc_test = np.mean(test_results[dict_value])
         logging.info("Window "+str(dict_value)+" mean acc:"+str(mean_acc_test))
    for dict_value in train_results.keys():
        logging.info('train:')
        mean_acc_train = np.mean(train_results[dict_value])
        logging.info("Window "+str(dict_value) +" mean acc:"+ str(mean_acc_train))
    
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
    with open(args.save+'/'+'outcomes_'+ file_name[:-3] + idx_ +'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
         pickle.dump([train_results, test_results, etime], f)

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 2.1
The primary function to load the dataset.
"""    
def call_ldataset(fold,clas,pic_):
    resetv()
    for i in range(len(clas)):
        ldataset(fold+clas[i], i, pic_)

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 2.1
This function calls the script that loads the dataset 
and starts the training process
"""
def run_tr(fl_):
    global dataset, labels, names, last_column, first_column, numfiles
    resetv()
    file_path = "../../data/turbulent_gas_mixtures/"
    if not names:
        with open(file_path+fl_ + '.pkl', 'rb') as f_s:
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
def lauch(clas_,p):
    fold_='dataset_twosources_downsampled/'
    print('loading ' + fold_) 
    #The next line could be omitted if the dataset is preloaded
    #call_ldataset(fold_,clas_,p) #Execute this line to load the dataset  
    run_tr('preloaded_dataset')
    
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SECTION 1.
The script begins here. 
"""
global num_classes
clas_=['Et_n', 'Et_L', 'Et_M', 'Et_H']  #Classes

#Added By Ismael
num_classes = len(clas_)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

args.save = ('{}-{}-TurbulentGasMixtures').format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

lauch(clas_, pic_) # loading the dataset
