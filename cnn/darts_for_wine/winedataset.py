import torch
from torch.utils.data.dataset import Dataset
"""
Created on Tue Dec 5 21:10:04 2018
@author: Ismael Cesar
e-mail: ismael.c.s.a@hotmail.com
"""
import torch
from winesC20 import calload



class WinesDataset(Dataset):
    """
        Class to be used by the data loader for getting the data.
    """

    def __init__(self,ds_names):
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
                data = []
                labels = []
                labels_ = []
                names = []
                iterations = 0
                for ds in ds_names:
                    r_data, r_lbls, r_lbls_, r_names = calload(self.array_measurements,self.pic_,ds,self.load)
                    if iterations == 0:
                        data.append(r_data)
                        labels.append(r_lbls)
                        labels_.append(r_lbls_)
                        names.append(r_names)
                        iterations += 1
                    else:
                        data    += r_data[0]
                        labels  += r_lbls[0]
                        labels_ += r_lbls_[0]
                        names   += r_names[0]

                self.data = torch.FloatTensor(data)       #Creating a tensor out of the data loaded
                self.labels = torch.FloatTensor(labels)   #Creating a tensor out of the labels loaded
                self.labels_ = torch.FloatTensor(labels_) #Creating a tensor out of the labels loaded
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
        return self.data[item], self.labels[item], self.labels_[item]

    def __len__(self):
        """
        :return: returns the size of the dataset
        """
        return len(self.data)