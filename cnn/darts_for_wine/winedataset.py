"""
Created on Tue Dec 5 21:10:04 2018
@author: Ismael Cesar
e-mail: ismael.c.s.a@hotmail.com

"""
import numpy as np
import torch
from torch.utils.data.dataset    import Dataset
from darts_for_wine.winesC20 import calload



class WinesDataset(Dataset):
    """
        Class to be used by the data loader for getting the data.
    """

    def __init__(self,chosen_ds_dict):
        """
        :param:chosen_ds_dict -> A dict with the following parameters:{
                                'array_measurements':array with te number of measurements of each bottle,
                                'pic_': Set to 1 if desired to use pictures and 0 if otherwise,
                                'opt':'TR',
                                'load':Set to 1 if the data still needs to be loaded, 0 otherwise,
                                'class_number': number of classes,
                                'procedure': procedure to be used when loading the data set
                                }
        """
        if(type(chosen_ds_dict) == dict):

            """
            After verifying if each element of the list is of type string
            the constructors sets the parameter for loading the data
            """
            self.array_measurements = chosen_ds_dict['array_measurements']
            self.pic_ = chosen_ds_dict['pic_']
            self.load = chosen_ds_dict['load']
            self.opt = chosen_ds_dict['opt']
            # auxilary variable for holding the data loaded in the pkl
            data    = None
            labels  = None
            labels_ = None
            names   = None
            iterations = True

            #r stands for read.
            #r_data, r_lbls, r_lbls_, r_names = calload(self.array_measurements,self.pic_,file_path,ds,self.load)
            r_data, r_lbls, r_lbls_, r_names = chosen_ds_dict['procedure'](self.array_measurements,self.pic_,
                                                                           self.opt,self.load)
            #For each sample in the data set create a four dimentional tensor with the shape [1,n_elements,lines,columns]
            for r_d,r_lbs,r_lbs_,r_nms in zip(r_data,r_lbls,r_lbls_,r_names):
                if iterations:
                    data    = torch.FloatTensor([[r_d]])
                    labels = [r_lbs]
                    labels_ = [r_lbs_]
                    names = [r_nms]
                    iterations = False
                else:
                    data = torch.cat((data, torch.FloatTensor([[r_d]])), dim=0)
                    #labels = torch.cat((labels, torch.Tensor(r_lbs)), dim=0)
                    labels.append(r_lbs)
                    #labels_ = torch.cat((labels, torch.Tensor(r_lbs_)), dim=0)
                    labels_.append(r_lbs_)
                    names.append(r_nms)

            self.data = torch.FloatTensor(data)       #Creating a tensor out of the data loaded
            self.labels = torch.Tensor(labels)   #Creating a tensor out of the labels loaded
            self.labels_ = torch.Tensor(labels_) #Creating a tensor out of the labels loaded
            self.names = names
        else:
            raise Exception("The element in the constructor must be a dict")

    def __getitem__(self, item):
        """
        :param item:
        :return: A tuple with the tensor and its corresponding labels
        """
        return (self.data[item],self.labels[item])

    def __len__(self):
        """
        :return: returns the size of the dataset
        """
        return self.data.shape[0]
