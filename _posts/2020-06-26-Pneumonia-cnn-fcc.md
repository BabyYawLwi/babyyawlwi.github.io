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
Firstly, the required modules are imported:  
`
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
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt  
%matplotlib inline
`
The dataset and test dataset will be loaded first from the folders and  the images in the datasets are transformed by resizing to (224,224) and then normalizing.
`
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
`  

We will split the 10% of the dataset into validation dataset and the rest as training dataset.  
`
val_pct = 0.1
val_size = int(val_pct * len(dataset))
train_size = len(dataset) - val_size
train_data, valid_data = random_split(dataset, [train_size, val_size])
`

Next, we define training, validation and testing data loaders for retrieving images in batches.
`
BATCH_SIZE = 64
train_dl = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,pin_memory=True)
valid_dl = DataLoader(valid_data,batch_size=BATCH_SIZE*2,num_workers=2,pin_memory=True)
test_dl = DataLoader(test_data,batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
`

### Previewing Images
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
![image showing single chest x-ray image with label]({{ site.baseurl }}/images/single-x-ray.png "single chest x-ray image with label") 

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
![batch of chest x-ray images]({{ site.baseurl }}/images/batch-x-ray.png "batch of chest x-ray images") 

### Configuring GPU

```
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
        
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)
        
device = get_default_device()

trn_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(valid_dl, device)
tst_dl = DeviceDataLoader(test_dl, device)
```

### Model

```
def accuracy(outputs, labels):
    preds = [1 if pred>0.5 else 0 for pred in outputs]
    preds = torch.as_tensor(preds, dtype=torch.float32, device=device)
    preds = preds.view([torch.tensor(preds.shape).item(), 1])
    return torch.tensor(torch.sum(preds == labels).item() / len(preds), device=device)
    
class BinaryClassificationBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch 
        targets = torch.tensor(targets.clone().detach(), dtype=torch.float32, device=device)
        targets = targets.view([torch.tensor(targets.shape).item(),1])
        out = self(images)
        loss = F.binary_cross_entropy(out, targets) 
        return loss
    
    def validation_step(self, batch):
        images, targets = batch 
        targets = torch.tensor(targets.clone().detach(), dtype=torch.float32, device=device)
        targets = targets.view([torch.tensor(targets.shape).item(),1]) 
        out = self(images)
        loss = F.binary_cross_entropy(out, targets)
        acc = accuracy(out, targets)
        return {'val_loss': loss.detach(), 'val_acc': acc }
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_scores = [x['val_acc'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()      
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_score.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
            
class PneumoniaCnnModel(BinaryClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        n_features = self.network.fc.in_features
        self.network.fc = nn.Linear(n_features, 1)
    
    def forward(self, xb):        
        return torch.sigmoid(self.network(xb))      
        
model = to_device(PneumoniaCnnModel(), device)        
```

### Training the Model


```
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            sched.step()
        
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history
    
history = [evaluate(model, val_dl)]

num_epochs = 10
max_lr = 1e-2
opt_func = torch.optim.Adam
grad_clip = 0.1
weight_decay = 1e-4

history += fit(num_epochs, max_lr, model, trn_dl, val_dl, weight_decay=weight_decay, grad_clip=grad_clip, opt_func=opt_func)
```

### Making Predictions on Test Data

```
@torch.no_grad()
def predict_dl(dl, model):
    torch.cuda.empty_cache()
    batch_probs = []
    for xb, _ in tqdm(dl):
        probs = model(xb)
        batch_probs.append(probs.cpu().detach())
    batch_probs = torch.cat(batch_probs)
    return batch_probs
    
test_predictions = predict_dl(tst_dl, model)
test_accuracy = accuracy(test_predictions, test_labels)    
```

### Saving the trained model and its parameters

```
torch.save(model.state_dict(), 'chest-x-ray-resnet50-model.pth')

import jovian
jovian.log_hyperparams(arch='resnet50-imgnet', 
                       epochs=num_epochs, 
                       lr=max_lr, 
                       scheduler='one-cycle',
                       weight_decay=weight_decay,
                       grad_clip=grad_clip,
                       opt=opt_func.__name__)

jovian.log_metrics(train_loss=history[-1]['train_loss'],
                   val_loss=history[-1]['val_loss'],
                   val_accuracy=history[-1]['val_acc'],
                   test_accuracy=test_accuracy.item())

jovian.commit(project=project_name, environment=None, outputs=['chest-x-ray-resnet50-model.pth'])
```




































