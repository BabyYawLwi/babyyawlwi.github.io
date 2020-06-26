---
layout: post
title: Classification of Chest X-ray Images for Pneumonia using Transfer Learning in Pytorch
---
### Introduction
This is my first project on deep learning (especially Convolutional Neural Networks) with Pytorch. In this project, I will be using the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). The dataset is initially organized into 3 folders such as 'train','val' and 'test'. Each contains subfolders for each category such as 'Pneumonia' and 'Normal'.
The code snippets below are from Jupyter Notebook which you can find on my [GitHub](https://www.github.com/babyyawlwi)   
![image showing normal and pneumonia chest x-rays]({{ site.baseurl }}/images/chest-x-ray-image.png "Chest x-ray images")  

### Organizing the dataset
The training folder contains 1341 Normal and 3875 Pneumonia Chest X-ray Images; and the testing folder contains 234 Normal and 390 Pneumonia images. The validation folder, however, contains only 8 images for each category. Instead of using the validation folder, I decided to use 10% of training data as validation data.  
> Firstly, the required modules are imported:  
```
import os    
import torch   
import pandas as pd  
import numpy as np  
from torch.utils.data import Dataset, random_split, DataLoader  
from PIL import Image  
import torchvision.models as models  
import torchvision.transforms as transforms  
from torchvision import datasets  
import torch.nn.functional as F  
import torch.nn as nn  
from torchvision.utils import make_grid  
import matplotlib.pyplot as plt  
%matplotlib inline
```
The dataset and test dataset will be loaded first from the folders and  the images in the datasets are transformed by resizing to (224,224) and then normalizing.
```
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_transforms = transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(*stats,inplace=True)
                                      ])
test_transforms = transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST),
                                      transforms.ToTensor(),
                                      transforms.Normalize(*stats)
                                     ])
dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
test_data = datasets.ImageFolder(TEST_DIR, transform=test_transforms)                                   
```  

We will split the 10% of the dataset into validation dataset and the rest as training dataset.  
```
val_pct = 0.1
val_size = int(val_pct * len(dataset))
train_size = len(dataset) - val_size
train_data, valid_data = random_split(dataset, [train_size, val_size])
```

Next, we define training, validation and testing data loaders for retrieving images in batches.
```
BATCH_SIZE = 64
train_dl = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,pin_memory=True)
valid_dl = DataLoader(valid_data,batch_size=BATCH_SIZE*2,num_workers=2,pin_memory=True)
test_dl = DataLoader(test_data,batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
```

### Explore and Preview Images
```
def decode_label(label_number):
    if label_number==0:
        return "NORMAL"
    return "PNEUMONIA"
    
def show_image(img_tuple):
    plt.imshow(img_tuple[0].permute(1,2,0))
    print("Label: ",decode_label(img_tuple[1]))
show_image(dataset[0])
```
![image showing single chest x-ray image with label]({{ site.baseurl }}/images/single-x-ray-image.png "single chest x-ray image with label") 

```
def show_batch(dl, invert=False):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xticks([]); ax.set_yticks([])
        data = 1-images if invert else images
        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))
        break
show_batch(train_dl)        
```
![batch of chest x-ray images]({{ site.baseurl }}/images/batch-x-ray-image.png "batch of chest x-ray images") 




















