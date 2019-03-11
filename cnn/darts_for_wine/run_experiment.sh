#!/bin/bash
#Use this Script in order to run an experiments sequence 

#Running experiments using fonollosa dataset
python fonollosa3.py --data_set_option 5
python fonollosa3.py --data_set_option 4
python fonollosa3.py --data_set_option 3
python fonollosa3.py --data_set_option 2
python fonollosa3.py --data_set_option 1

#Running experiments QWines data set
#python LoadQWinesCSystem.py --data_set_option 1 --is_using_wine_ds True
#python LoadQWinesEaCSystem.py --data_set_option 1 --is_using_wine_ds True