"""
Created on Tue Dec 5 21:10:04 2018
@author: Ismael Cesar
e-mail: ismael.c.s.a@hotmail.com
"""
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

import cnn.utils as utils
from darts_for_wine.winedataset import WinesDataset
from model_search import Network
from architect import Architect


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

global CLASSES_WINE

def run_experiment_darts_wine():

    files_path = "../../data/wines/"
    #ds_names = ["QWines-CsystemTR", "QWinesEa-CsystemTR"]
    ds_names = ([["QWines-CsystemTR"],3],[["QWinesEa-CsystemTR"],4])
    #epochs  = args.epochs
    learning_rate = args.learning_rate
    CLASSES_WINE = ds_names[0][1]

    criterion = nn.CrossEntropyLoss()
    #criterion.cuda()

    model = Network(args.init_channels,CLASSES_WINE,args.layers,criterion)
    #model.cuda()

    #Adam was chosen Because is fast and converges with good stability
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=args.weight_decay
    )

    ds_wine = WinesDataset(files_path,ds_names[0][0])
    logging.info("The data set has been loaded")

    ds_lenght  = len(ds_wine)
    ds_indices = range(ds_lenght)
    ds_split   = -1
    train_queue = torch.utils.data.DataLoader(ds_wine,sampler=torch.utils.data.SubsetRandomSampler(ds_indices[ds_split:ds_lenght]),
                                              batch_size=args.batch_size)

    valid_queue = torch.utils.data.DataLoader(ds_wine, sampler=torch.utils.data.SubsetRandomSampler(ds_indices[:ds_split]))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    #architecht = Architect(model,args)
    #The loop has also been adapted to the Leave one out technique
    for epoch in  range(args.epochs):
        scheduler.step()

        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        F.softmax(model.alphas_normal, dim=-1)
        F.softmax(model.alphas_reduce, dim=-1)

        #Reusing the train procedure of the DARTS implementation
        train_acc, train_obj = train(train_queue,model,criterion,optimizer,CLASSES_WINE)
        logging.info('train_acc %f', train_acc)


    valid_acc, valid_obj = infer(valid_queue,model,criterion)
    logging.info('valid_acc %f', valid_acc)

    file_index = 1
    utils.save(model,os.path.join(args.save,"wine_"+str(file_index)+".pt"))

"""
The train e infer procedure were addapted for the leave one out technique
"""

def train(train_queue, model,criterion,optimizer,num_classes):
  """
    :param train_queue: Data loader that randomly picks the samples in the Dataset, as defined in the previous procedure
    :param valid_queue: Data loader that randomly picks the samples in the Dataset, as defined in the previous procedure
    :param model:       Model of the network
    :param criterion:   Criterion(Function over which the loss of the model shall be computed)
    :param optimizer:  weights optimizer
    :param lr: learning rate
    :return: train_acc(train accuracy), train_obj(Object used to compute the train accuracy)
  """
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False)#.cuda()
    target = Variable(target, requires_grad=False)#.cuda(async=True)

    # get a random minibatch from the search queue with replacement
    #input_search, target_search = next(iter(valid_queue))
    #input_search = Variable(input_search, requires_grad=False)#.cuda()
    #target_search = Variable(target_search, requires_grad=False)#.cuda(async=True)
    #architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, torch.LongTensor([target]))

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, torch.LongTensor([target]), topk=(1, num_classes - 1))
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)
    #objs.update(loss.data[0], n)
    #top1.update(prec1.data[0], n)
    #top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
  """
  :param valid_queue: Data loader that randomly picks the samples in the Dataset, as defined in the previous procedure
  :param model:      Model of the network
  :param criterion:  Criterion(Function over which the loss of the model shall be computed)
  :return: valid_acc(validation accuracy), valid_obj(Object used to compute the validation accuracy)
  """
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

if __name__ == "__main__":
    run_experiment_darts_wine()