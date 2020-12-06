#-*- coding: utf-8 -*-


import random
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


###Python Random Seed 고정###
SEED = 1234 # 원하는 seed값을 사용하시면 됩니다.

random.seed(SEED) # python에서 random 한 부분을 해당 seed값으로 고정합니다.
torch.manual_seed(SEED) # torch에서 random한 부분을 해당 seed값으로 고정합니다.
torch.cuda.manual_seed(SEED) # torch의 cuda연산에서 random한 부분을 해당 seed값으로 고정합니다.

###----------------------###



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
  
  # nn.CrossEntropyLoss()를 사용하는 경우, train 코드에서
  # labels = labels.to(device, dtype=long) 또는,
  # labels = labels.to(device).long() 과 같은 방법으로 data type을 변경하여 사용해 보세요.

  def __len__(self):
    return len(self.data)



### train 또는 test dataset에 대하여, num의 수만큼 subplot을 보여주는 함수입니다.
def image_show(dataset, num):
  fig = plt.figure(figsize=(30,30))

  for i in range(num):
    plt.subplot(1, num, i+1)
    plt.imshow(dataset[i][0].squeeze())
    plt.title(dataset[i][1])
