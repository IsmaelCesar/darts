"""
Created on Tue Dec 5 21:10:04 2018
@author: Ismael Cesar
e-mail: ismael.c.s.a@hotmail.com

"""
import numpy as np
import torch
from torch.utils.data.dataset    import Dataset
#from darts_for_wine.winesC20 import calload



class WinesDataset(Dataset):
    """
        Class to be used by the data loader for getting the data.
    """

    def __init__(self,data,labels):
        self.data = torch.from_numpy(data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])).float()
        self.labels = torch.from_numpy(labels).float()
        

    def __getitem__(self, item):
        """
        :param item:
        :return: A tuple with the tensor and its corresponding labels
        """
        return (self.data[item].cuda(),self.labels[item].cuda())

    def __len__(self):
        """
        :return: returns the size of the dataset
        """
        return self.data.shape[0]
