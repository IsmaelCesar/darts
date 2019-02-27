import os
import csv
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

class PerclassAccuracyMeter(object):
    """
    Class dedicated for bulding the list to the CSV file
    """
    def __init__(self,num_classes):

        csv_list = [['train_acc']]
        valid_acc_added = 0
        while valid_acc_added < 2:
            for i in  range(num_classes):
                csv_list[0] += ["class"+str(i)+"_acc"]


            valid_acc_added += 1

            if valid_acc_added == 1:
                csv_list[0] += ["valid_acc"]

        csv_list.append(np.zeros(num_classes * 2 + 2).tolist())
        self.csv_list = csv_list
        self.current_epoch = 0
        self.num_classes = num_classes
        self.first_iteration = False
        self.reset_perclass_params()

    def compute_perclass_accuracy(self,taget,predictions,batch_size,epoch,is_train=True):
        """
        :param taget: Target value(s) from the Data set
        :param predictions:  Predictions made by the model
        :param batch_size:   Batch size
        :param epoch:        Learning epoch
        :param offset:       skipping train_acc e valid_acc values
        :return:             CSV list updated
        """

        _,indexes = torch.max(predictions,1)
        if self.current_epoch < epoch:
            self.csv_list.append(np.zeros(self.num_classes*2+2).tolist())
            self.current_epoch = epoch

        offset = 1
        offset_perclass_params = 0

        if not is_train :
            offset = self.num_classes + 2
            offset_perclass_params = self.num_classes

        for idx,tgt in zip(indexes,taget):
            if idx == tgt:
                self.perclass_acc_cont[offset_perclass_params+idx] += 1
                self.perclass_avg_cont[offset_perclass_params+idx] += 1

                self.perclass_acc[offset_perclass_params+idx] = (self.perclass_acc_cont[offset_perclass_params+idx]/
                                                                 batch_size)

                self.perclass_sum[offset_perclass_params+idx] += self.perclass_acc[offset_perclass_params+idx]

        for sum,n,i in zip(self.perclass_sum,self.perclass_avg_cont,range(self.num_classes*2)):
            if n > 0:
                if i == 4:
                    self.csv_list[epoch+1][i+2] = sum/n *100
                else:
                    self.csv_list[epoch + 1][i + 1] = sum / n * 100

        self.perclass_acc_cont = np.zeros(self.num_classes * 2).tolist()

        return self.csv_list

    def write_csv(self,file_path, mode="a+"):
        if (self.first_iteration):
            mode = 'w+'
            with open(file_path, mode) as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                csv_writer.writerows(self.csv_list)
                csv_file.close()
        else:
            new_list = self.csv_list[1:]
            with open(file_path, mode) as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                csv_writer.writerows(new_list)
                csv_file.close()

    def reset_perclass_params(self):
        self.perclass_sum = np.zeros(self.num_classes * 2).tolist()
        self.perclass_avg = np.zeros(self.num_classes * 2).tolist()
        self.perclass_acc_cont = np.zeros(self.num_classes * 2).tolist()
        self.perclass_avg_cont = np.zeros(self.num_classes * 2).tolist()
        self.perclass_acc = np.zeros(self.num_classes * 2).tolist()

    def include_top1_avg_acc(self,top1,is_train=True):
        """
        :param top1: Value that includes top1 prediction average accuracy
        """
        offset = 0
        if not is_train:
          offset = self.num_classes+1

        self.csv_list[self.current_epoch+1][offset] = top1

    def return_current_epoch_data(self):
        string_value = ""
        for v in ['train_acc','valid_acc']:
            string_value += "\n"+v + ": \n"
            offset = 1

            if v == 'valid_acc':
                offset = 2 + self.num_classes

            for c in range(self.num_classes):
                 string_value += "class_"+str(c+1)+"acc: "
                 string_value += str(self.csv_list[self.current_epoch+1][offset+c]) + " \n"

        return string_value



def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()

  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

