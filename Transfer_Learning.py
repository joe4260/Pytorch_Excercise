# 라이브러리 호출
import os
import time
import copy
import glob
import cv2
import shutil

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

#이미지 데이터 전처리 방법 정의
data_path = '../chap05/data/catanddog/train'

transform = transforms.Compose(
  [
    transforms.Resize([256, 256]),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
  ])
train_dataset = torchvision.datasets.ImageFolder(
  data_path,
  transform = transform
)
train_loader= torch.utils.data.DataLoader(
  train_dataset,
  batch_size = 32,
  num_workers = 8,
  shuffle = True
)

print(len(train_dataset))
