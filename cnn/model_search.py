import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

class MixedOp(nn.Module):
  """
    Module contains the implementations defined in the operations.py module
  """
  def __init__(self, C, stride):
    """
    :param C:Number of channels
    :param stride: convolution stride
    """
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    """
      Given the PRIMITIVES operations defined in the module genotype.py.
      Includes the implementation of each operation to the self._ops variable.
    """
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    """
    :param x: input
    :param weights: alphas
    :return: a summation of each operation applied in the input and pondered by its corresponding alpha
    """
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    """
    :param steps: steps of foward propagation of a computation cell
    :param multiplier: auxiliary variable to create other convolution layers
    :param C_prev_prev: Number of channels of the layer that comes before the prevoius layer
    :param C_prev: Number of channels of the previous layer
    :param C: Number of channels of the current layer
    :param reduction: flag that mark this computation cell as a cell to be reduced
    :param reduction_prev: flag of the previous computation call
    """
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    """
      Iterates over the number or feedfoward steps, and for each step 
      creates a MixedOp module.
    """
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    """
    :param s0: input 1
    :param s1: input 2
    :param weights: alphas
    :return:
    """
    s0 = self.preprocess0(s0)#apply first preprocessor to the first input
    s1 = self.preprocess1(s1)#apply second preprocessor to the second input

    states = [s0, s1]
    offset = 0
    """
      Iterates over the computation steps and applies the inputs to MixedOp added in the creation
      of the module with the alpha, according to some offset.
      Offset indicate what operation shall be applied over what input(s0 or s1), two by two
    """
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
  """
    Module implements a network over which the search shall be performed,
  """

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    """
    :param C: Number of channels
    :param num_classes:
    :param layers:
    :param criterion: The optimizer to update the weights
    :param steps: steps of foward propagation of a computing cell
    :param multiplier: Auxiliary variable to create the first convolution layer
    :param stem_multiplier: Auxiliary variable to create the other convolutions
    """
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      #The number of input channels have been changed from 3 to C
      nn.Conv2d(C, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    """
      Loop iterates over the number of layers to create the inner computation
      cells, markint the ones in the interval [layers//3, 2*layers//3] as cells to be reduced.
      Which means tha at most the first three layers and the last two shall not be reduced.
      For each layer a computation cell is created.
    """
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion)#.cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)#Applies the convolution and batch norm of the firs layer
    """
      Interates over the computations cells in the network.
      Applying the computed ouputs of each one using the weights(alphas) 
    """
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    """
    :param input:
    :param target:
    :return: tensor with the loss function computed according to the optmizer defined in the
              constructor of the class
    """
    logits = self(input)
    #logits.squeeze_()
    #target.squeeze_()
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    """
      Randomly initializes alphas, based on the number of operations described in the genotype module.
    """
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)
    #cuda()
    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):
    """
      :return: corresponding genotype of the network
    """

    def _parse(weights):
      """
      :param weights: alphas that where reduced during the search
      :return: genotype corresponding to the set of alphas reduced.
      """
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

