#-*- coding: utf-8 -*-


import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class TypeData(Dataset):
  '''
### Digit일 경우 label로 0을, ###
### Letter일 경우 label로 1을 ###
### return하는 class입니다. ###

사용 예시:
train_data = TypeData(train=True)
test_data = TypeData(train=False)
  '''
  def __init__(self,train):
    super(TypeData, self).__init__()
    self.digit = 10
    self.letter = 46
    self.train = train

    self.data = torchvision.datasets.EMNIST(root='./',
                                        split='bymerge',
                                        train=self.train,
                                        transform=transforms.ToTensor(),
                                        download=True)

  def __getitem__(self, index):
    if self.data[index][1] < self.digit:
      label = 0.
    else:
      label = 1.
    return self.data[index][0], label

  def __len__(self):
    return len(self.data)



### train 또는 test dataset에 대하여, num의 수만큼 subplot을 보여주는 함수입니다.
def image_show(dataset, num):
  fig = plt.figure(figsize=(30,30))

  for i in range(num):
    plt.subplot(1, num, i+1)
    plt.imshow(dataset[i][0].squeeze())
    plt.title(dataset[i][1])
