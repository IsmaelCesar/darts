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

    def __init__(self, data, labels, use_nparrays=False):
        # Transform the numpy arrays into float tensors
        """
            :param data: Array with the data
            :param labels: An array with the corresponding labels
            :param use_nparrays: if set to false. the procedure will transform the nparrays into
                                pytorch tensors
        """
        self.data = []
        self.labels = []
        if use_nparrays == False:

            for d, l in zip(data, labels):
                self.data.append(d.reshape(1, d.shape[0], d.shape[1]).tolist())
                self.labels.append(l)

            self.data = torch.FloatTensor(self.data)
            self.labels = torch.IntTensor(self.labels)

        else:
            self.data = torch.from_numpy(data.reshape(data.shape[0], 1, data.shape[1], data.shape[2]))
            self.labels = torch.from_numpy(labels)

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
