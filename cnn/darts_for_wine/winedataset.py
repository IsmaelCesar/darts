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

    def __init__(self,file_path,ds_names):
        """
        :param:ds_names -> A list of strigns containing the name of the data sets
                            it can be a list containing just one name. But it has to be a lists
        """
        if(type(ds_names) == list):
            if(type(el) == str for el in ds_names):
                """
                After verifying if each element of the list is of type string
                the constructors sets the parameter for loading the data
                """
                self.array_measurements = [4,5,13]
                self.pic_ = 0
                self.load = 0
                # auxilary variable for holding the data loaded in the pkl
                data    = None
                labels  = None
                labels_ = None
                names   = None
                iterations = True
                for ds in ds_names:
                    #r stands for read.
                    r_data, r_lbls, r_lbls_, r_names = calload(self.array_measurements,self.pic_,file_path,ds,self.load)
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
                raise Exception("The element in the constructor must be list of strings")
        else:
            raise Exception("The element in the constructor must be list of strings")

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
