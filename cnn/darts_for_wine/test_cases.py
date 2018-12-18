import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as  np
from cnn.darts_for_wine.winedataset import WinesDataset

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


if __name__ == '__main__':
    test_data_loading()