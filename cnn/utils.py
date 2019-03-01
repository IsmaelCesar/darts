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
    in the csv list there shal be the accuracy per-class
    along with the perclass error.
    """
    def __init__(self,num_classes):

        csv_list = [['train_acc']]
        valid_acc_added = 0
        while valid_acc_added < 2:
            for i in  range(num_classes):
                csv_list[0] += ["class"+str(i)+"_acc"]
                csv_list[0] += ["class"+str(i)+"_error_rate"]


            valid_acc_added += 1

            if valid_acc_added == 1:
                csv_list[0] += ["valid_acc"]

        csv_list.append(np.zeros(num_classes * 4 + 2).tolist())
        self.csv_list = csv_list
        self.current_epoch = 0
        self.num_classes = num_classes
        self.first_iteration = False
        self.reset_confusion_matrix()


    def compute_confusion_matrix(self,taget,logits):

        _, preds = torch.max(logits,1)

        for t,p in zip(taget.view(-1),preds.view(-1)):
            self.confusion_matrix[t,p] += 1

    def reset_confusion_matrix(self):
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)

    def compute_perclass_accuracy(self,epoch,is_train=True):

        perclass_acc = self.confusion_matrix.diag() / self.confusion_matrix.sum(1)
        perclass_error = self.__compute_perclass_error()
        #Adding the values to the csv_list
        if self.current_epoch < epoch:
            self.current_epoch = epoch
            self.csv_list.append(np.zeros(self.num_classes*4+2).tolist())

        offset = 1

        if not is_train:
            offset = self.num_classes*2 + 2

        for p_acc,pc_error,i in zip(perclass_acc,perclass_error,range(0,self.num_classes*2,2)):
            self.csv_list[self.current_epoch+1][offset+i] = p_acc.item()*100
            self.csv_list[self.current_epoch+1][offset+i+1] = pc_error.item()

        return perclass_acc

    def __compute_perclass_error(self):
        """
        The error computed in this method is the error rate of each class.
        Let fp - false positives, tp - true positives, fn - false negatives
        and tn - true negatives

        For each class the following equation is computed
        (fp + fn) /(tp + tn + fp + fn)

        :return: list of perclass t2 error
        """
        perclass_error = []

        for i in range(self.num_classes):
            fp =0
            fn =0
            for j in range(self.num_classes):
                if i!=j:
                    fp += self.confusion_matrix[i][j]
                    fn += self.confusion_matrix[j][i]
            result = (fp + fn) / (self.confusion_matrix.diag().sum().item() + fp + fn)

            perclass_error.append(result)

        return perclass_error

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
                offset = 2 + self.num_classes*2
            c = 0
            for i in range(0,self.num_classes*2,2):
                 string_value += "class_"+str(c)+"acc: "
                 string_value += str(self.csv_list[self.current_epoch+1][offset+i]) + "\t\t"
                 string_value += "class_" + str(c) + "_error_rate: "
                 string_value += str(self.csv_list[self.current_epoch + 1][offset + i + 1]) + "\n"
                 c+= 1

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

