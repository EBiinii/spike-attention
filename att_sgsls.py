import time
import timm
import gc
import torch


# In[1]:

import cv2
import os
import re
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import copy

# from tensorboardX import SummaryWriter
from torchvision import datasets, transforms


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.optim as optim
from torchvision.transforms.functional import to_pil_image

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# torch.cuda.set_per_process_memory_fraction(0.9, device=0)  # 예시로 0.9로 설정
# allocated_memory = torch.cuda.memory_allocated()
# print(f"현재 할당된 CUDA 메모리: {allocated_memory / 1024 / 1024} MB")
# reserved_memory = torch.cuda.memory_reserved()
# print(f"예약된 CUDA 메모리: {reserved_memory / 1024 / 1024} MB")
# In[34]:
gc.collect()
torch.cuda.empty_cache()

data_path = './data'
checkpoint_dir = './checkpoint'
checkpoint_name = 'att-sgsls-30'

epoch = 50
batch_size = 8
num_classes = 5
shape = (300,300,3)

learning_rate = 0.001
weight_decay = 0.0001


# In[35]:


cutmix_alpha = 1
cutmix_prob = 0.3


# In[36]:


if not os.path.isdir(data_path):
  os.makedirs(data_path)
if not os.path.isdir(os.path.join(checkpoint_dir, checkpoint_name)):
  os.makedirs(os.path.join(checkpoint_dir, checkpoint_name))


# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
gc.collect()
torch.cuda.empty_cache()

# # Randaugment

# In[38]:


import random

# import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageDraw


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return ImageOps.autocontrast(img)


def Invert(img, _):
    return ImageOps.invert(img)


def Equalize(img, _):
    return ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = Image.fromarray(imgs[i])
        return Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


def augment_list():  # 16 oeprations and their ranges

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        # (AutoContrast, 0, 1),
        # (Equalize, 0, 1),
        # (Invert, 0, 1),
        (Rotate, 0, 30),
        # (Posterize, 0, 4),
        # (Solarize, 0, 256),
        # (SolarizeAdd, 0, 110),
        # (Color, 0.1, 1.9),
        # (Contrast, 0.1, 1.9),
        # (Brightness, 0.1, 1.9),
        # (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        # (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    return l



class RandAugment(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
            # mask = op(mask, val)
        # mask = ImageOps.grayscale(mask)
        return img

# In[39]:
from corrupt import *
import imagenetC as IC

val_transform = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])


basic_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    RandAugment(3, 2),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-15,15)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])

rand_transform = transforms.Compose([
    transforms.Resize(100),
    corrupt({"gaussian_noise": 0, "shot_noise": 0,"impulse_noise": 0,"speckle_noise": 0,
             "gaussian_blur": 0,"glass_blur": 0,"defocus_blur": 0,"zoom_blur": 0,
             "motion_blur": 0.5}),
    # RandAugment(3, 2),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-15,15)),
    transforms.ToTensor()
])
# rand_transform.transforms.insert(0, RandAugment(3, 2))

data_path = './data/train/'
#data augmentation

train_dataset = torchvision.datasets.ImageFolder(
    data_path,
    transform=basic_transform
)


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# print(len(train_dataset))

test_path = './data/test/'

# transform = transforms.Compose(
#                 [
#                     transforms.Resize([224, 224]),
#                     transforms.ToTensor(),
#                 ])
test_dataset = torchvision.datasets.ImageFolder(
    root=test_path,
    transform=val_transform
)

val_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
print(len(train_dataset), len(test_dataset))

# In[ ]:

# # 데이터셋 잘 만들어졌는지 확인
# for i in range(10):
#   train_features, train_labels = next(iter(train_loader))
#   img = train_features[0].squeeze()
#   img = img.permute(1,2,0)
#   label = train_labels[0]

#   img_n = checkpoint_dir + '/' + checkpoint_name + '/img_' + str(i) + '_' +str(label) + '.png'
#   plt.imshow(img)
#   ax = plt.gca()
#   ax.axis('off')
#   # ax.axes.xaxis.set_visible(False)
#   # ax.axes.yaxis.set_visible(False)
#   ax.axes.get_xaxis().set_visible(False)
#   ax.axes.get_yaxis().set_visible(False)

#   plt.savefig(img_n,dpi=300, bbox_inches='tight', pad_inches=0)
# #   plt.show()
#   print(f"Label: {label}")



# In[ ]:


from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):


    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


# In[ ]:


import torch.optim.lr_scheduler as lrs
def make_scheduler(optimizer):
  cosine_scheduler = lrs.CosineAnnealingLR(
            optimizer,
            T_max=epoch
  )
  scheduler = GradualWarmupScheduler(
      optimizer,
      multiplier=1,
      total_epoch=epoch//10,
      after_scheduler=cosine_scheduler
  )
  return scheduler


# # utils

# In[ ]:


import matplotlib
# matplotlib.use('Agg')
def plot_learning_curves(metrics, cur_epoch):
    x = np.arange(cur_epoch+1)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ln1 = ax1.plot(x, metrics['train_loss'], color='tab:red')
    ln2 = ax1.plot(x, metrics['val_loss'], color='tab:red', linestyle='dashed')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ln3 = ax2.plot(x, metrics['train_acc'], color='tab:blue')
    ln4 = ax2.plot(x, metrics['val_acc'], color='tab:blue', linestyle='dashed')
    lns = ln1+ln2+ln3+ln4
    plt.legend(lns, ['Train loss', 'Validation loss', 'Train accuracy','Validation accuracy'])
    plt.tight_layout()
    plt.savefig('{}/{}/learning_curve.png'.format(checkpoint_dir, checkpoint_name), bbox_inches='tight')
    plt.close('all')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()


    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # print(batch_size)
    # print(pred)
    # print(correct)

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# In[ ]:


# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools


# In[ ]:


# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 30
print(torch.cuda.device_count())

# # Module

# In[ ]:


import torch
import torchvision.ops
from torch import nn

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x, offset):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(offset)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x


# In[ ]:


class ATT(nn.Module):
  def __init__(self, outc):
    super(ATT, self).__init__()

    self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(outc, outc, kernel_size=1, stride=1, padding=0, groups = outc),
            nn.BatchNorm2d(outc),
            nn.GELU(),
            nn.Conv2d(outc, outc, kernel_size=1, stride=1, padding=0, groups = outc),
        )
    self.spatial_att1 = nn.Conv2d(outc, outc, kernel_size=1, stride=1, padding=0, groups = outc)
    self.spatial_att2 = nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1, groups = outc)
    self.spatial_att3 = nn.Conv2d(outc, outc, kernel_size=5, stride=1, padding=2, groups = outc)
    self.conv_f = nn.Conv2d(outc*3, outc, kernel_size=1, stride = 1, padding=0)

    self.act = nn.GELU()
    self.conv1 = nn.Conv2d(outc, outc, kernel_size=1, stride = 1, padding=0)

    self.sigmoid = nn.Sigmoid()

    self.IN = nn.InstanceNorm2d(outc, affine=True)
    self.deform_conv1 = DeformableConv2d(outc, outc, kernel_size=1, stride=1, padding=0, bias=True)
    self.softmax = nn.Softmax()
    self.gap = nn.AdaptiveAvgPool2d(1)

    self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

  def forward(self, gf):
    mem1 = self.lif1.init_leaky()
    mem2 = self.lif2.init_leaky()
    mem3 = self.lif3.init_leaky()

    spk1, mem1 =self.lif1(gf, mem1)

    # gf = self.act(self.conv1(gf))
    gf_c = self.global_att(spk1)
    spk2, mem2 = self.lif2(gf_c, mem2)

    gf_c = self.sigmoid(spk2)
    
    gf_c = gf * gf_c    

    global_feat1 = self.spatial_att1(gf_c)
    global_feat2 = self.spatial_att2(gf_c)
    global_feat3 = self.spatial_att3(gf_c)
    global_feat = torch.cat([global_feat1, global_feat2, global_feat3], dim=1)

    att = self.act(self.conv_f(global_feat))

    att = self.deform_conv1(att, att)
    # att2 = dcn * att
    
    wei = self.softmax(att)
    
    atts = gf_c * wei
    cls = self.IN(atts)
    spk3, mem3 = self.lif3(cls, mem3)
    
    # cls = cls + gf


    return spk3



# In[ ]:


import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.residual_function(x) + self.shortcut(x))

class ResNetSNN(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.att1 = ATT(64)
        self.att2 = ATT(128)
        self.att3 = ATT(256)
        self.att4 = ATT(512)


    def _make_layer(self, block, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        mem_fc = torch.zeros(512,13,13).cuda()

        cur1 = self.conv1(x)
        # cur1 = self.att1(cur1)

        for t in range(num_steps):

          cur2 = self.conv2_x(cur1)
          spk2 = self.att1(cur2)

          cur3 = self.conv3_x(spk2)
          spk3 = self.att2(cur3)

          cur4 = self.conv4_x(spk3)
          spk4 = self.att3(cur4)

          cur5 = self.conv5_x(spk4)
          spk5 = self.att4(cur5)

          mem_fc = mem_fc + spk5


        out_voltage = mem_fc / num_steps
        # out_voltage = self.att4(out_voltage)
        output = self.avg_pool(out_voltage)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18(num_classes=10, **kargs):
    """ return a ResNet 18 object
    """
    return ResNetSNN(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# In[ ]:

 
model = resnet18(num_classes=5).to(device)
# x = torch.randn(2, 3, 100, 100).to(device)
# pred = model(x)
# print(f"output shape: {pred.shape}")


# In[ ]:
gc.collect()
torch.cuda.empty_cache()
from torchsummary import summary

summary(model, (3,100,100))



# In[ ]:


# # Trainer-classification

# In[ ]:


class Trainer:
    def __init__(self, model, criterion1, optimizer, scheduler):
        self.model = model
        self.criterion1 = criterion1
        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.attacker = attacker

    def train(self, data_loader, epoch, result_dict):
        total_loss = 0
        count = 0
        since = time.time()

        losses = AverageMeter()
        top1 = AverageMeter()

        self.model.train()

        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            # inputs_adv = self.attacker(inputs, labels)

            cls = self.model(inputs)

            loss = self.criterion1(cls, labels)

            if len(labels.size()) > 1:
                labels = torch.argmax(labels, axis=1)

            prec1, prec3 = accuracy(cls.data, labels, topk=(1, 3))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward(retain_graph = False) #메모리와 속도 차이 원래 True
            self.optimizer.step()

            total_loss += loss.tolist()
            count += labels.size(0)


            _s = str(len(str(len(data_loader.sampler))))
            ret = [
                ('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count, len(data_loader.sampler), 100 * count / len(data_loader.sampler)),
                'train_loss: {: >4.2e}'.format(total_loss / count),
                'train_accuracy : {:.2f}%'.format(top1.avg),
            ]
            print(', '.join(ret))

        self.scheduler.step()
        result_dict['train_loss'].append(losses.avg)
        result_dict['train_acc'].append(top1.avg)
        
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        result_dict['train_time'].append(time_elapsed)

        return result_dict


# In[ ]:


import json

class Evaluator():
    def __init__(self, model, criterion1):
        self.model = model
        self.criterion1 = criterion1

    def worst_result(self):
        ret = {
            'loss': float('inf'),
            'accuracy': 0.0
         }
        return ret

    def result_to_str(self, result):
        ret = [
            'epoch: {epoch:0>3}',
            'loss: {loss: >4.2e}'
        ]
        for metric in self.evaluation_metrics:
            ret.append('{}: {}'.format(metric.name, metric.fmtstr))
        return ', '.join(ret).format(**result)

    def save(self, result):
        with open(self.save_path, 'w') as f:
            f.write(json.dumps(result, sort_keys=True, indent=4, ensure_ascii=False))

    def load(self):
        result = self.worst_result
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                try:
                    result = json.loads(f.read())
                except:
                    pass
        return result

    def evaluate(self, data_loader, epoch, result_dict):
        losses = AverageMeter()
        top1 = AverageMeter()
        since = time.time()

        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.cuda(), labels.cuda()
                # inputs_adv = self.attacker(inputs, labels)

                cls = self.model(inputs)

                loss = self.criterion1(cls, labels)

                prec1, prec3 = accuracy(cls.data, labels, topk=(1, 3))

                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
        
        time_elapsed = time.time() - since
        print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        result_dict['test_time'].append(time_elapsed)

        print('----Validation Results Summary----')
        print('Epoch: [{}] Top-1 accuracy: {:.2f}%'.format(epoch, top1.avg))

        result_dict['val_loss'].append(losses.avg)
        result_dict['val_acc'].append(top1.avg)

        return result_dict


# In[ ]:


from torch import optim as optim
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


# DataParallel로 모델을 감싸기
model = nn.DataParallel(model)
# criterion1 = BCEDiceLoss()
criterion1 = LabelSmoothingCrossEntropy(smoothing=0.1)
# criterion3 = nn.MSELoss()
# criterion2 = nn.CrossEntropyLoss()
# criterion2 = FocalLoss()
from adabelief_pytorch import AdaBelief
parameters = model.parameters()
# optimizer = AdaBelief(model.parameters(), lr=1e-3, eps=1e-8, betas=(0.9,0.95), weight_decouple = True, rectify = False)
optimizer = optim.AdamW(parameters, lr = 1e-3, betas=(0.9, 0.95))
# optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.000deform_conv2d1)
# optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
# optimizer = make_optimizer(model)
scheduler = make_scheduler(optimizer)
result_dict = {'epoch':[], 'train_loss' : [], 'train_acc' : [], 'train_iou' : [], 'train_dsc' : [],'train_recon' : [],
               'val_loss' : [], 'val_acc' : [], 'val_iou' : [], 'val_dsc' : [], 'val_recon' : [], 'train_time' : [], 'test_time' : []}
# trainer = Trainer(model, criterion1, criterion2, criterion3, optimizer, scheduler, attacker)
# evaluator = Evaluator(model, criterion1, criterion2, criterion3, attacker)

trainer = Trainer(model, criterion1, optimizer, scheduler)
evaluator = Evaluator(model, criterion1)


# In[ ]:


# evaluator.save(result_dict)

best_val_acc = 0.0
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name, 'checkpoint.pth')
torch.manual_seed(7)
""" define training loop """
for n_epoch in range(epoch):
    result_dict['epoch'] = n_epoch
    result_dict = trainer.train(train_loader, n_epoch, result_dict)
    result_dict = evaluator.evaluate(val_loader, n_epoch, result_dict)

    if result_dict['val_acc'][-1] >= best_val_acc:

        ret = [
            ('epoch: {: } ').format(n_epoch),
            'train_acc : {:.3f}'.format(result_dict['val_acc'][-1])
        ]
        print(', '.join(ret) )
        # print("{} epoch, best epoch was updated! {}%".format(n_epoch, result_dict['val_acc'][-1]))
        best_val_acc = result_dict['val_acc'][-1]
        # model.save(checkpoint_name='best_model')
        torch.save(model.state_dict(), checkpoint_path)


# In[ ]:


print(best_val_acc)
train_time_minutes = sum(result_dict['train_time']) // 60
train_time_seconds = sum(result_dict['train_time']) % 60
print('Total train time in {:.0f}m {:.0f}s'.format(train_time_minutes, train_time_seconds))
test_time_minutes = sum(result_dict['test_time']) // 60
test_time_seconds = sum(result_dict['test_time']) % 60
print('Total test time in {:.0f}m {:.0f}s'.format(test_time_minutes, test_time_seconds))


# In[ ]:


plot_learning_curves(result_dict, n_epoch)


# # 모델 평가
# 

# In[ ]:


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()


    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # print(batch_size)
    # print(pred)
    # print(correct)

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# In[ ]:



checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name, 'checkpoint.pth')
model.load_state_dict(torch.load(checkpoint_path))
top1 = AverageMeter()

model.eval()
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(val_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)

        prec1, prec3 = accuracy(outputs.data, labels, topk=(1, 3))
        top1.update(prec1.item(), inputs.size(0))

print('----Test Set Results Summary----')
print('Top-1 accuracy: {:.2f}%'.format(top1.avg))


# In[ ]:


y_pred = []
y_true = []

model.eval()
for batch_idx, (inputs, labels) in enumerate(val_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth
# # confusion matrix

# In[ ]:

class_list = ['0', '1', '2', '3','4']
from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
print(confusion_matrix(y_true , y_pred))
print('Classification Report')
print(classification_report(y_true , y_pred, target_names=class_list))

# %matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name, 'cm.png')
    plt.savefig(checkpoint_path)



# cm_plot_labels = class_list

# plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


# In[ ]:



plot_confusion_matrix(cm=cm, classes=class_list, title='Confusion Matrix')

# In[ ]:
from sklearn.manifold import TSNE
def visualize_tsne(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actual = []
    deep_features = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = model(images)

            deep_features += features.cpu().numpy().tolist()
            actual += labels.cpu().numpy().tolist()
        
    # TSNE 적용
    tsne = TSNE(n_components=2, random_state=0)
    cluster = np.array(tsne.fit_transform(np.array(deep_features)))
    actual = np.array(actual)

    # 시각화
    plt.figure(figsize=(5, 5))
    name = [0, 1, 2, 3, 4, 5]
    for i, label in zip(range(5), name):
        idx = np.where(actual == i)
        plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=label)

    plt.legend()
    # plt.show()
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name, 'tsne.png')
    plt.savefig(checkpoint_path)

visualize_tsne(model, val_loader)
# In[ ]:


y_pred = np.array(y_pred)
y_true = np.array(y_true)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y_true)
encoded_Y = encoder.transform(y_true)

from keras.utils import to_categorical
# y_true = np_utils.to_categorical(encoded_Y)
y_true = to_categorical(encoded_Y)

# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y_pred)
encoded_Y = encoder.transform(y_pred)

from keras.utils import to_categorical
# y_pred = np_utils.to_categorical(encoded_Y)
y_pred = to_categorical(encoded_Y)


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
from sklearn.metrics import roc_auc_score


# Plot linewidth.
lw = 2
n_classes = 5

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
        label='micro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
# plt.show()
# plt.savefig('roc.png'.format(checkpoint_dir, checkpoint_name), bbox_inches='tight')
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name, 'roc.png')
plt.savefig(checkpoint_path)

# Grad_Cam #####################################################
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM, AblationCAM, RandomCAM
from pytorch_grad_cam.metrics.road import ROADCombined

import warnings
warnings.filterwarnings('ignore')
from torchvision import models
import numpy as np
import cv2
import requests
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from PIL import Image

# print(model)
model = resnet18(num_classes=5).cuda()
model.eval()

img_path = './data/grad.jpg'
img = Image.open(img_path)
resize_transform = transforms.Resize((100, 100))
img = resize_transform(img)
img = np.float32(img) / 255
input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# 입력 텐서를 GPU로 이동합니다.
input_tensor = input_tensor.cuda()
# The target for the CAM is the Bear category.
# As usual for classication, the target is the logit output
# before softmax, for that category.
targets = [ClassifierOutputTarget(4)]
target_layers = model.conv5_x

with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])
images = np.hstack((np.uint8(255*img), cam , cam_image))
Image.fromarray(images)

cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
scores = cam_metric(input_tensor, grayscale_cams * 0, targets, model)
print(f"Empty CAM, Combined metric avg confidence increase with ROAD accross 4 thresholds (positive is better): {scores[0]}")

from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget

# For the metrics we want to measure the change in the confidence, after softmax, that's why
# we use ClassifierOutputSoftmaxTarget.
targets = [ClassifierOutputSoftmaxTarget(4)]
cam_metric = CamMultImageConfidenceChange()
scores, visualizations = cam_metric(input_tensor, grayscale_cams, targets, model, return_visualization=True)
score = scores[0]
visualization = visualizations[0].cpu().numpy().transpose((1, 2, 0))
visualization = deprocess_image(visualization)
print(f"The confidence increase percent: {100*score}")
print("The visualization of the pertubated image for the metric:")
Image.fromarray(visualization)

# Showing the metrics on top of the CAM : 
def visualize_score(visualization, score, name, percentiles):
    visualization = cv2.putText(visualization, name, (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    # visualization = cv2.putText(visualization, "(Least first - Most first)/2", (10, 40), 
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
    # visualization = cv2.putText(visualization, f"Percentiles: {percentiles}", (10, 55), 
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)    
    # visualization = cv2.putText(visualization, "Remove and Debias", (10, 70), 
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA) 
    visualization = cv2.putText(visualization, f"{score:.5f}", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)    
    return visualization
    
def benchmark(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False, category=4):  # 클래스 인덱스 수정
    methods = [("GradCAM", GradCAM(model=model, target_layers=target_layers)),
               ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers)),
               ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers)),
               ("AblationCAM", AblationCAM(model=model, target_layers=target_layers)),
               ("RandomCAM", RandomCAM(model=model, target_layers=target_layers))]

    cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
    targets = [ClassifierOutputTarget(category)]
    metric_targets = [ClassifierOutputSoftmaxTarget(category)]
    result_paths = []
    visualizations = []
    vis = []
    percentiles = [10, 50, 90]
    for name, cam_method in methods:
        with cam_method:
            attributions = cam_method(input_tensor=input_tensor, 
                                      targets=targets, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
        attribution = attributions[0, :]    
        scores = cam_metric(input_tensor, attributions, metric_targets, model)
        score = scores[0]
        
        vis = show_cam_on_image(img, attribution, use_rgb=True)
         # Save each visualization individually
        result_path = os.path.join(checkpoint_dir, checkpoint_name, f'{name}_grad_cam.png')
        # result_paths.append(result_path)
        Image.fromarray(vis).save(result_path)

        visualization = visualize_score(vis, score, name, percentiles)
        visualizations.append(visualization)  

    # Combine visualizations horizontally
    combined_image = np.hstack(visualizations)
    return Image.fromarray(combined_image)

result_image = benchmark(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False)

checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name, 'grad_cam.png')
result_image.save(checkpoint_path)
