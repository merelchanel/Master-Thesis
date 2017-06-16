### Code used for the experimental part in Phyton. 

First importing the libraries. 
```
# Import libraries
import pandas as pd
import numpy as np
import scipy 

import matplotlib.pyplot as plt
import random
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve
from sklearn.metrics import classification_report, cohen_kappa_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
```

Then, the data is imported using pandas, and the targetlabel is seperated. 
```
# Import data pandas
pdhost = pd.read_csv("finaldata.csv")

# Delete first column
pdhost = pdhost.drop(['Unnamed: 0'], axis = 1)

# Create array for labels
y = pdhost['chosen'].values

# Delete labels from dataframe
pdhost = pdhost.drop('chosen', axis = 1) 
```

We convert the categorical variables to dummies and make the columns sparse. 
```
# Convert categorical values to dummies and convert to numpy array
# The parameter Sparse=True is used to make the columns sparse. 
X = pd.get_dummies(pdhost, sparse=True).values
X.shape
```
Checking the shape at this time: 38452 instances, 50 features. 

Then we split the data into train, validation and test splits. 
```
# Create a train and testsplit
from sklearn.model_selection import train_test_split

X_rest, X_test, y_rest, y_test = \
   train_test_split(X, y, test_size=0.20, random_state=123)  # random seed makes this reproducible

print(X_rest.shape, y_rest.shape)
print(X_test.shape, y_test.shape)

# Create a validation split
X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=y_test.shape[0], random_state=123)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
```
The shapes of the final datasets are: 
X_train = (23070, 50)
X_val = (7691, 50)
X_test = (7691, 50)

Applying PCA is tested to see whether it improves results. 
```
from sklearn.decomposition import PCA

# Apply regular PCA
pca = PCA()
X = pca.fit_transform(X)
```
However, it did not improve the results and therefore the results are not used. 

The function 'report' is created to easily obtain all our metrics 
