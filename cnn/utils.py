import os
import csv
import numpy as np
import torch
import shutil
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
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
    def __init__(self, num_classes):

        main_train_metrics_header = [['epoch', 'train_acc', 'train_loss']]
        secondary_train_metrics_header = [['epoch']]
        valid_acc_added = 0

        for i in  range(num_classes):
            secondary_train_metrics_header[0] += ["class_"+str(i)+"_train_precision"]
            secondary_train_metrics_header[0] += ["class" + str(i) + "_train_recall"]
            secondary_train_metrics_header[0] += ["class" + str(i) + "_train_f1_score"]

        main_valid_metrics_header = [['valid_acc', 'valid_loss']]
        secondary_valid_metrics_header = []

        for i in range(num_classes):
            secondary_valid_metrics_header += ["class" + str(i) + "_valid_precision"]
            secondary_valid_metrics_header += ["class" + str(i) + "_valid_recall"]
            secondary_valid_metrics_header += ["class" + str(i) + "_valid_f1_score"]

        secondary_valid_metrics_header = [secondary_valid_metrics_header]
    
        self.main_train_metrics_header = main_train_metrics_header
        self.main_train_metrics_values = []

        self.secondary_train_metrics_header = secondary_train_metrics_header
        self.secondary_train_metrics_values = []

        self.main_valid_metrics_header = main_valid_metrics_header
        self.main_valid_metrics_values = []

        self.secondary_valid_metrics_header = secondary_valid_metrics_header
        self.secondary_valid_metrics_values = []
        self.num_classes = num_classes

    def __write_main_metrics_csv(self, file_path, mode='w+'):

        temp_header = [self.main_train_metrics_header[0] + self.main_valid_metrics_header[0]]
        temp_values = np.concatenate([self.main_train_metrics_values, self.main_valid_metrics_values], axis=1).tolist()
        csv_list = temp_header + temp_values

        with open(file_path, mode) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerows(csv_list)
            csv_file.close()

    def __write_secondary_metrics_csv(self, file_path, mode='w+'):

        temp_header = [self.secondary_train_metrics_header[0] + self.secondary_valid_metrics_header[0]]
        temp_values = np.concatenate([self.secondary_train_metrics_values, self.secondary_valid_metrics_values],
                                     axis=1).tolist()
        csv_list = temp_header + temp_values

        with open(file_path, mode) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerows(csv_list)
            csv_file.close()

    def write_csv(self,file_path, main_metrics_file, secondary_metrics_file=None,
                  mode="w+", write_secondary_metrics=True):

        self.__write_main_metrics_csv(os.path.join(file_path, main_metrics_file), mode=mode)

        if write_secondary_metrics and secondary_metrics_file is not None:
            self.__write_secondary_metrics_csv(os.path.join(file_path, secondary_metrics_file), mode=mode)
        else:
            raise Exception("Secondary metrics file must not be None type")

    def compute_metrics(self, y_true, y_predicted):
        """
        Compute mestrics concerning Precision, Recall and F1_score per learning epoch
        :return: Precision, Recall and F_score
        """
        y_true_np = y_true
        if type(y_true) == torch.Tensor:
            y_true_np = y_true.cpu().detach().numpy()

        y_predicted_np = y_predicted.cpu().detach().numpy()
        y_predicted_np = np.argmax(y_predicted_np, axis=1)

        precision = precision_score(y_true_np, y_predicted_np, average=None).tolist()
        recall = recall_score(y_true_np, y_predicted_np, average=None).tolist()
        f_score = f1_score(y_true_np, y_predicted_np, average=None).tolist()

        return precision, recall, f_score

    def save_train_metrics_into_list(self, epoch, accuracy, loss, precision, recall, fscore):
        """
        Save computed metrics, including accuracy and loss into the train list
        :param epoch: current epoch
        :param accuracy: model's accuracy in current epoch
        :param loss: model's accuracy in current epoch
        :param precision: model's accuracy in current epoch
        :param recall: model's accuracy in current epoch
        :param f1_score: model's accuracy in current epoch# Computing other trainning metrics
        :return: void
        """
        temp_list = []

        for i in range(self.num_classes):
            temp_list += [precision[i], recall[i], fscore[i]]

        self.secondary_train_metrics_values.append([epoch] + temp_list)
        self.main_train_metrics_values.append([epoch, accuracy, loss])

    def save_validation_metrics_into_list(self, accuracy, loss, precision, recall, fscore):
        """
        Save computed metrics, including accuracy and loss into the train list
        :param accuracy: model's accuracy in current epoch
        :param loss: model's accuracy in current epoch
        :param precision: model's accuracy in current epoch
        :param recall: model's accuracy in current epoch
        :param f1_score: model's accuracy in current epoch
        :return: void
        """
        temp_list = []

        for i in range(self.num_classes):
            temp_list += [precision[i], recall[i], fscore[i]]

        self.secondary_valid_metrics_values.append(temp_list)
        self.main_valid_metrics_values.append([accuracy, loss])

    def display_current_epoch_metrics(self):
        """
        Display the computed metrics of the current epoch,
        Concerning the values in the csv_list_train and csv_list_valid elements of this object
        :param epoch:
        :return: String value of the current epoch metrics
        """

        string_value = ""

        # Displaying Train Metrics

        curent_epoch_metrics_train = self.secondary_train_metrics_values[-1]
        # Offset for skipping epoch cell
        offset_epoch = 1
        curent_epoch_metrics_train = curent_epoch_metrics_train[offset_epoch:]
        offset = 2
        string_value += "Train metrics: \n\n"
        for i in range(self.num_classes):
            string_value += "class_" + str(i) + "_precision: %.2f \t\t" % curent_epoch_metrics_train[offset-2]
            string_value += "class_" + str(i) + "_recall: %.2f    \t\t" % curent_epoch_metrics_train[offset-1]
            string_value += "class_" + str(i) + "_f1_score: %.2f  \t\t" % curent_epoch_metrics_train[offset]
            offset += 3
            string_value += "\n"

        string_value += "\n\n"

        # Displaying Valid Metrics
        curent_epoch_metrics_valid = self.secondary_valid_metrics_values[-1]
        # curent_epoch_metrics_valid = curent_epoch_metrics_valid

        offset = 2
        string_value += "Validation metrics: \n\n"
        for i in range(self.num_classes):
            string_value += "class_" + str(i) + "_precision: %.2f \t\t" % curent_epoch_metrics_valid[offset-2]
            string_value += "class_" + str(i) + "_recall: %.2f    \t\t" % curent_epoch_metrics_valid[offset-1]
            string_value += "class_" + str(i) + "_f1_score: %.2f \t\t" % curent_epoch_metrics_valid[offset]
            string_value += "\n"
            offset += 3

        string_value += "\n\n"
        return string_value


class SensorDataTransformer(object):
    """
    This object serves as a data transformer of the
    sensor measurements, saving values relevant to the data an canvas size
    """
    def __init__(self,canvas_lenght,canvas_height,max_lenght,max_height):

        self.canvas_lenght = canvas_lenght
        self.canvas_height = canvas_height
        self.max_lenght = max_lenght
        self.max_height = max_height
        self.dot = 255

    def __dot(self,canvas,d1,d2):
        if d1.any() < self.canvas_height:
            val1 = len(canvas) - round(d1 * self.canvas_height / self.max_height) - self.dot
            val2 = len(canvas) - round(d1 * self.canvas_height / self.max_height)
            val3 = np.floor_divide(d2 * self.canvas_lenght, self.max_lenght) - 1
            val4 = np.floor_divide(d2 * self.canvas_lenght, self.max_lenght) - 1 + self.dot
            canvas[int(val1)
                   :int(val2),
                val3:
                val4] = 255

        return canvas

    def tranform_sensor_values_to_image(self,sensor_array):
        """
        :param sensor_array: Array of samples to be tranformed in to images
        :param max_height: the maximum values of the measurements, related do maximum voltage
                            of the sensor
        :param max_lenght: the maximum values of the measurents, related to measurement time
        :return:
        """
        sensor_images = []
        for sample, j in zip(sensor_array,range(sensor_array.shape[0])):
            image = np.full((sensor_array.shape[2], self.canvas_lenght,self.canvas_height),0)
            for line in sample:
                for sensor, sensor_index in zip(line, range(len(line))):
                    image[sensor_index] = self.__dot(image[sensor_index], sensor, j)
            sensor_images.append(image)

        return np.array(sensor_images)


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

