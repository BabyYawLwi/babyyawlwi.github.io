---
layout: post
title: Credit Fraud Detection using SMOTE and GridSearch Pipeline
---
## Introduction
In order to prevent charging the customers for the transactions from imposters, detecting which transactions are fraud or not becomes vital for credit card companies. The aim of the project is to build a classifier to identify the fraudulent credit card transactions.  

## About dataset
This project uses the Credit Card Fraud Detection dataset from [kaggle.com](https://www.kaggle.com/mlg-ulb/creditcardfraud). The dataset contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where there were a total of 492 frauds out of 284,807 transactions. The dataset is highly unbalanced with the positive class (frauds) account for only 0.172% of all transactions.
<br>
It contains only numerical input variables which are the result of a PCA transformation. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. <br>
Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise. 
```
# Importing the libraries
import numpy as np
import pandas as pd
import os
import cv2
import time
import itertools
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, f1_score, precision_recall_curve, precision_score 
from sklearn.metrics import recall_score, average_precision_score, auc

import imblearn
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
```
https://jovian.ml/babyyawlwi/credit-fraud-pub-kaggle/v/3&cellId=5

## Light Gradient Boosting Model (LightGBM)
LightGBM is a gradient boosting framework that uses tree based learning algorithms. LightGBM model is trained on the highly imbalanced training dataset. It is designed to be distributed and efficient with the following advantages:
- Faster training speed and higher efficiency: Light GBM use histogram based algorithm i.e it buckets continuous feature values into discrete bins which fasten the training procedure.
- Lower memory usage: Replaces continuous values to discrete bins which result in lower memory usage.
- Better accuracy than any other boosting algorithm: It produces much more complex trees by following leaf wise split approach rather than a level-wise approach which is the main factor in achieving higher accuracy. However, it can sometimes lead to overfitting which can be avoided by setting the max_depth parameter.
- Compatibility with Large Datasets: It is capable of performing equally good with large datasets with a significant reduction in training time.
- Parallel learning supported.[1] <br>
```
classifier.fit(X_train, y_train)
ypred = classifier.predict(X_test)
```
https://jovian.ml/babyyawlwi/credit-fraud-pub-kaggle/v/3&cellId=26 <br>
As expected, the LightGBM model predicts the more populated non-fraud transactions very accurately but performs badly for fraud. There are 39 false negatives (which are actually frauds but predicted as non-frauds by the model) which amount to about 39 percent of the total positives. For the case of detecting credit card frauds, lowering the number of false negatives is the most vital performance to improve so as to meet the objective of preventing as many frauds as possible. 

## Resampling with SMOTE and GridSearchCV Pipeline
The model performance on the imbalanced dataset is not satisfactory so one of the approaches to address this problem is to oversample the minority class which is the fraud class. Synthetic Minority Oversampling Technique (SMOTE) is one of the oversampling techniques that creates synthetic minority class samples. SMOTE first selects a minority class instance a at random and finds its k nearest minority class neighbors. The synthetic instance is then created by choosing one of the k nearest neighbors b at random and connecting a and b to form a line segment in the feature space. The synthetic instances are generated as a convex combination of the two chosen instances a and b [2]
A Pipeline is constructed to search for the best parameters of the model using GridSearchCV while trained on the resampled data. 
```
rs_parameters = {
    'lgbmclassifier__learning_rate': [0.005,0.01,0.001,0.05],
    'lgbmclassifier__n_estimators': [20,40,60,80,100],
    'lgbmclassifier__num_leaves': [6,8,12,16]
    }
kf = KFold(n_splits=10, random_state=0, shuffle=True)
sampler = SMOTE(random_state=0)
smp_pipeline = make_pipeline(sampler, classifier)
grid_imba = GridSearchCV(smp_pipeline,
                         param_grid=rs_parameters,
                         cv=kf,
                         scoring='roc_auc',
                         return_train_score=True,
                         n_jobs=-1,
                         verbose=True
                        )
grid_imba.fit(X_train, y_train)
bestimator = grid_imba.best_estimator_
ypred = bestimator.predict(X_test)
```

## Conclusion
After resampling with SMOTE and optimizing the parameters using GridSearch, the LightGBM model shows distinct improvement. There is a drastic decrease in the number of false negatives for about 70% so more frauds are being detected by the model. Given the class imbalance ratio, it was recommended to measure the performance using the Area Under the Precision-Recall Curve (AUPRC). The AUPRC for the model with resampled data also increases compared to the model with imbalanced data. 

## References
[1] https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
[2] Page 47, Imbalanced Learning: Foundations, Algorithms, and Applications, 2013
