"""
Created on Tue Dec 5 21:10:04 2018
@author: Ismael Cesar
e-mail: ismael.c.s.a@hotmail.com
"""
import logging
import argparse
import torch

import cnn.utils as utils
from cnn.darts_for_wine.winedataset import WinesDataset
from cnn.model_search import Network
from cnn.architect import Architect
from cnn.train_search import train, infer

parser = argparse.ArgumentParser("DARTS for wine classification")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=1, help='num of init channels')#the initial channels of the data is one
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

CLASSES_WINE = 4

def run_experiment_darts_wine():

    files_path = "../../data/wines/"
    ds_names = ["QWines-CsystemTR", "QWinesEa-CsystemTR"]
    epochs  = args.epochs
    learning_rate = args.learning_rate

    criterion = torch.nn.CrossEntropyLoss
    #criterion.cuda()

    model = Network(args.init_channels,CLASSES_WINE,args.layers,criterion)
    #model.cuda()

    #Adam was chosen Because is fast and converges with good stability
    optmizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=args.weight_decay
    )

    ds_wine = WinesDataset(files_path,ds_names)
    logging.info("The data set has been loaded")

    ds_lenght  = len(ds_wine)
    ds_indices = range(ds_lenght)
    ds_split   = -2
    train_queue = torch.utils.data.DataLoader(ds_wine,sampler=torch.utils.data.SubsetRandomSampler(ds_indices[ds_split:ds_lenght]),
                                              batch_size=args.batch_size)

    valid_queue = torch.utils.data.DataLoader(ds_wine, sampler=torch.utils.data.SubsetRandomSampler(ds_indices[ds_split]))

    architecht = Architect(model,args)

    #Reusing the train procedure of the DARTS implementation
    train(train_queue,valid_queue,model,architecht,criterion,optmizer,learning_rate)



if __name__ == "__main__":
    run_experiment_darts_wine()