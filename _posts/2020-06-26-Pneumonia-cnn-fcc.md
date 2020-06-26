---
layout: post
title: Classification of Chest X-ray Images for Pneumonia using Transfer Learning in Pytorch
---
### Introduction
This is my first project on deep learning (especially Convolutional Neural Networks) with Pytorch. In this project, I will be using the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). The dataset is initially organized into 3 folders such as 'train','val' and 'test'. Each contains subfolders for each category such as 'Pneumonia' and 'Normal'.
The code snippets below are from Jupyter Notebook which you can find on my [GitHub](https://www.github.com/babyyawlwi)   
![image showing normal and pneumonia chest x-rays]({{ site.baseurl }}/images/chest-x-ray-image.png "Chest x-ray images")  

### Organizing the image dataset
The training folder contains 1341 Normal and 3875 Pneumonia Chest X-ray Images; and the testing folder contains 234 Normal and 390 Pneumonia images. The validation folder, however, contains only 8 images for each category. Instead of using the validation folder, I decided to use 10% of training data as validation data.  
Firstly, the required modules are imported:  
```import os  
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
