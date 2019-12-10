import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as  np
from darts_for_wine.winedataset import WinesDataset
from matplotlib.pyplot import imshow
from utils import PerclassAccuracyMeter

class AnotherNet(nn.Module):
    def __init__(self):
        super(AnotherNet,self).__init__()
        layers      = nn.Sequential()
        layers.add_module("layer_1",nn.Linear(10,20))
        layers.add_module("relu_1",nn.ReLU())
        layers.add_module("layer_2",nn.Linear(20,20))
        layers.add_module("relu_2", nn.ReLU())
        layers.add_module("layer_3",nn.Linear(20,20))
        layers.add_module("relu_2", nn.ReLU())
        layers.add_module("layer_4",nn.Linear(20,10))
        layers.add_module("relu_3", nn.ReLU())
        self.net_layers = layers

    def forward(self,x):
        return self.net_layers.forward(x)

def main():
    print("Using DARTS for wine classification")
    batch_size = 10
    values = torch.randn(batch_size,batch_size)
    target = torch.randn(batch_size)
    resultingLoss = None
    mNetwork  = AnotherNet()
    criterion = nn.MSELoss()
    for i in range(batch_size):
      logit =  mNetwork(values[i])
      mNetwork.zero_grad()
      loss  =  criterion(logit,target[i])
      resultingLoss = loss
      print(loss)
      loss.backward()
    torch.save(mNetwork.state_dict(),"weights.pt")

def carrega_modelo():
    print("Carregando modelo")
    mNetwork = AnotherNet()
    print("Parametros: \n",list(mNetwork.parameters()))
    mNetwork.load_state_dict(torch.load("weights.pt"))
    mNetwork.eval()
    print("Parametros modelo carregado:\n ", list(mNetwork.parameters()))


def carrega_dados_vinho():
    print("Carregando dados para ser rodado no modelo")
    data_files = ["QWines-CsystemTR.pkl","QWinesEa-CsystemTR.pkl"]
    dir_path = "../"
    with  open(dir_path+data_files[0],'rb') as f:
        ds,lbls,lbls_,names = pickle.load(f)
        newDs = torch.FloatTensor(ds)
        print(newDs.shape)
        print(len(ds[0]))

def test_data_loading():
    ds_names = ["QWines-CsystemTR","QWinesEa-CsystemTR"]
    m_data  = WinesDataset(ds_names)
    batch_size = 10
    data_size  = len(m_data)
    indices    = range(data_size)
    train_queue = torch.utils.data.DataLoader(m_data,sampler=torch.utils.data.SubsetRandomSampler(indices[1:]),
                                              batch_size=batch_size)

    for i, (data,target1,target2) in enumerate(train_queue):
        print("Shape of the data: \n",data.shape)
        print("Shape of the target1: \n", target1.shape)
        print("Shape of the target2: \n", target2.shape)

def testing_csv_list(perclass_meter):


    batch_size = 10

    #myCsv  = np.random.randn(10,num_classes*2+2)
    num_classes = perclass_meter.num_classes
    acc = 0
    loss = 100
    for epoch in range(10):

        end_slice   = batch_size
        #Train
        target = torch.randint(num_classes, (100,)).numpy().tolist()
        logits = torch.rand(100, num_classes)

        train_precision, train_recall, train_fscore = perclass_meter.compute_metrics(target, logits)

        acc += 0.1
        loss -= 10

        perclass_meter.save_train_metrics_into_list(int(epoch), acc, loss,
                                                    train_precision, train_recall, train_fscore)

        #validation
        val_labels = torch.randint(num_classes, (50,))
        val_logits = torch.rand(50, num_classes)

        val_precision, val_recall, val_fscore = perclass_meter.compute_metrics(val_labels, val_logits)
        perclass_meter.save_validation_metrics_into_list(acc, loss,
                                                         val_precision, val_recall, val_fscore)

        print(perclass_meter.display_current_epoch_metrics())
    perclass_meter.write_csv("", "per_epoch_loss_and_accuracy.csv", "per_epoch_precision_recall_fscore.csv")
    return perclass_meter.main_train_metrics_values, perclass_meter.main_valid_metrics_values

if __name__ == '__main__':
    num_classes = 10
    perclass_meter = PerclassAccuracyMeter(num_classes)

    train_h, valid_h = testing_csv_list(perclass_meter)
    print("Train h:", np.array(train_h)[:, 1].mean())
    print("valid h:", np.array(valid_h)[:, 0].mean())
