#!/bin/bash
#Use this Script in order to run an experiments sequence 

#Running experiments using fonollosa dataset
#python fonollosa3.py --data_set_option 5
#python fonollosa3.py --data_set_option 4
#python fonollosa3.py --data_set_option 3
#python fonollosa3.py --data_set_option 2
#python fonollosa3.py --data_set_option 1

#Running experiments QWines data set
#python LoadQWinesCsystem.py --data_set_option 1 --is_using_inner_epoch_loop  --epochs 50
#python LoadQWinesEaCsystem.py --data_set_option 1 --is_using_inner_epoch_loop --epochs 50
#python coffee_dataset.py --is_using_inner_epoch_loop --epochs 50

python loading_WTD_dataset_relative_.py --is_using_inner_epoch_loop --epochs 50
