"""
Created on Tue Dec 5 21:10:04 2018
@author: Ismael Cesar
e-mail: ismael.c.s.a@hotmail.com
"""
import os
import time
import sys
sys.path.append("../")
import logging
import argparse
import torch
import torch.utils.data as torchdata
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import utils
from darts_for_wine.winedataset import WinesDataset
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("DARTS for wine classification")
parser.add_argument('--data', type=str, default='../../data/wines/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=1, help='num of init channels')#the initial channels of the data is one
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP_DARTS_WINE', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()


args.save = 'search-{}-{}-LoadQWinesCsystem'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
global CLASSES_WINE, csv_list

utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save ,'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)




def run_experiment_darts_wine(train_data,train_labels,test_data,test_labels,epochs,classes_number,model,window_n):
    
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    

    logging.info("\n\t WINDOW + %s\n",window_n)

    csv_list = [['avg_train_acc','ata_standard_deviation','valid_acc','valid_stdd']]

    CLASSES_WINE =  classes_number

    dataset_files_path = args.data


    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    if(model == None):
        model = Network(args.init_channels,CLASSES_WINE,args.layers,criterion)
        model.cuda()
        logging.info("A new model has been created")

    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=args.weight_decay
    )
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_ds_wine = WinesDataset(train_data,train_labels)
    test_ds_wine  = WinesDataset(test_data,test_labels)

    logging.info("The data set has been loaded")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    ds_lenght = len(train_ds_wine)
    loo_test_element_indx = 0

    architecht = Architect(model,args)

    train_queue = torch.utils.data.DataLoader(train_ds_wine,sampler=torchdata.sampler.SequentialSampler(train_ds_wine),
                                              batch_size=args.batch_size,
                                              pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(test_ds_wine,sampler=torchdata.sampler.SequentialSampler(test_ds_wine),
                                              pin_memory=True, num_workers=2)

    #The STDD will be used to calculate the accuracy's standard deviation
    stdd     = utils.StandardDeviationMeter()

    #The loop has also been adapted to the Leave one out technique
    for epoch in  range(epochs):

        scheduler.step()

        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        F.softmax(model.alphas_normal, dim=-1)
        F.softmax(model.alphas_reduce, dim=-1)

        #Reusing the train procedure of the DARTS implementation
        train_acc, train_obj,train_stdd = train(train_queue,valid_queue,model,lr,architecht,criterion,optimizer,CLASSES_WINE)
        #train_acc, train_obj, train_stdd = 2.0,2.0,3.0
        test_acc, test_obj, test_stdd   = infer(valid_queue,model,criterion,CLASSES_WINE)

        csv_list.append([train_acc.item(),train_stdd.item(),test_acc.item(),test_stdd.item()])

    #Saving the model
    utils.write_csv(csv_list,os.path.join(args.save,"experiments_measurements.csv"))
    utils.save(model,os.path.join(args.save,"wine_classifier_"+str(window_n)+".pt"))

    return csv_list,model

"""
The train e infer procedure were addapted for the leave one out technique
"""

def train(train_queue,valid_queue, model,lr,architect,criterion,optimizer,num_classes):
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
  stddm = utils.StandardDeviationMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    #get a random minibatch from the search queue with replacement
    #input_search, target_search = next(iter(valid_queue))
    #input_search = Variable(input_search, requires_grad=False)#.cuda()
    #target_search = Variable(target_search, requires_grad=False)#.cuda(async=True)
    #architect.step(input,torch.LongTensor([target]), input_search, torch.LongTensor([target_search]), lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    # torch.cuda.LongTensor([target])
    loss = criterion(logits,torch.cuda.LongTensor([target]))
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    # torch.cuda.LongTensor([target])
    prec1, prec5 = utils.accuracy(logits, torch.cuda.LongTensor([target]), topk=(1, num_classes))
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)
    #objs.update(loss.data[0], n)
    #top1.update(prec1.data[0], n)
    #top5.update(prec5.data[0], n)
    stddm.add_value(top1.avg)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  #Salvando topk
  stddm.calculate()
  return top1.avg,objs.avg,stddm.standard_deviation,

def infer(valid_queue, model, criterion,num_classes):
  """
  :param valid_queue: Data loader that randomly picks the samples in the Dataset, as defined in the previous procedure
  :param model:      Model of the network
  :param criterion:  Criterion(Function over which the loss of the model shall be computed)
  :return: valid_acc(validation accuracy), valid_obj(Object used to compute the validation accuracy)
  """
  # Changing report freq
  #infer_report_freq = 1

  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  stddm = utils.StandardDeviationMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits = model(input)
    #torch.cuda.LongTensor([target])
    loss = criterion(logits, torch.cuda.LongTensor([target]))

    #torch.cuda.LongTensor([target])
    prec1, prec5 = utils.accuracy(logits, torch.cuda.LongTensor([target]), topk=(1, num_classes))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)
    stddm.add_value(top1.avg)
    # objs.update(loss.data[0], n)
    # top1.update(prec1.data[0], n)
    # top5.update(prec5.data[0], n)

    #if step % infer_report_freq == 0:
    logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  stddm.calculate()
  return top1.avg, objs.avg, stddm.standard_deviation

if __name__ == "__main__":
    run_experiment_darts_wine()
