---
layout: post
title: Credit Fraud Detection using SMOTE and GridSearch Pipeline
---
## Introduction

  
## About dataset
```
code
```


Next, we define training, validation and testing data loaders for retrieving images in batches.

  

## Model
We will then create a CNN model with resnetWe will then create a CNN model with resnet50 architecture. Since it is a classification problem, the binary cross entropy loss and accuracy are calculated. We will then transfer the model to the default device.50 architecture and transfer the model to the default device. 
```
code
```

## Training the Model
During the training of the model, we will use one-cycle learning rate policy to schedule the learning rate; weight decay as regularization technique and gradient clipping.
```
code
```

## Conclusion
According to the accuracy performance of the test dataset, there is more than 20% difference compared to the best training performance. It is evident that the model is more or less overfitted by the training data so it needs to be further regularized. In addition, since it is a classification problem, there is a threat for imbalanced data causing good accuracy so the model performances should be measured by more suitable metrics like F1 score or AUROC (area under ROC).


