### Code used for the experimental part in Phyton. 

First importing the libraries. 
```
# Import libraries
import pandas as pd
import numpy as np
import scipy 

from sklearn import linear_model
from sklearn import metrics
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
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
Checking the shape at this time: 38452 instances, 96 features. 
