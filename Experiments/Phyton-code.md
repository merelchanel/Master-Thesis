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
from sklearn.grid_search import GridSearchCV
from time import time
from sklearn.model_selection import GridSearchCV

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
- X_train = (23070, 50)
- X_val = (7691, 50)
- X_test = (7691, 50) 

Applying PCA is tested to see whether it improves results. 
```
from sklearn.decomposition import PCA

# Apply regular PCA
pca = PCA()
X = pca.fit_transform(X)
```
However, it did not improve the results and therefore the results are not used. 

Code to add feature interactions: 
```
# Feature intereactions 
from sklearn.preprocessing import PolynomialFeatures
inter = PolynomialFeatures(degree=2, interaction_only=True)
X_train_zi = inter.fit_transform(X_train)
X_val_zi = inter.transform(X_val)
```

The function 'report' is created to easily obtain all our metrics:
```
def report(y_val, y_val_pred):
    print("Accuracy")
    print(accuracy_score(y_val, y_val_pred))
    print("F1 score")
    print(f1_score(y_val, y_val_pred, average = 'weighted'))
    print("Confusion matrix: ")
    print(confusion_matrix(y_val, y_val_pred))
    print("Classification report:")
    print(classification_report(y_val, y_val_pred))
    print("Cohen's Kappa:")
    print(cohen_kappa_score(y_val, y_val_pred))
    print("AUC")
    print(roc_auc_score(y_val, y_val_pred))
    print("//// end of report ///")
```

Grid search to search for best parameters, with K-NN as example: 
```
# Set parameters
cf = KNeighborsClassifier()
param_grid = {"n_neighbors": [1,2,3,4,5,6,7,8,9,10,11,12,13,14]}

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


grid_search = GridSearchCV(cf, param_grid, scoring=score)
start = time()
grid_search.fit(X_train_zi, y_arr)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
```

Then the different classifiers are tested on the validationset, 
with the parameter settings that resulted best on the grid-search:

**KNN**
```
from sklearn.neighbors import KNeighborsClassifier

cf = KNeighborsClassifier()
param_grid = {"n_neighbors": [1,2,3,4,5,6,7,8,9,10,11,12,13,14]}

knn = KNeighborsClassifier(n_neighbors = 13)
knn.fit(X_train, y_train)
y_val_pred = knn.predict(X_val)
metrics(y_val, y_val_pred)
``` 
**Random Forest**
```
from sklearn.ensemble import RandomForestClassifier

# tested with class_weights = {1:(1 till 10)}

# Set parameters
cf = RandomForestClassifier()
param_grid = {"max_features": ['auto', 'sqrt', 0.2, 1, 3, 10], 
              "max_depth": [3, None],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

rf = RandomForestClassifier(bootstrap=False, min_samples_leaf=3, 
                            min_samples_split=2, criterion='gini', 
                            max_features=0.2, max_depth=None)
```
**Neural Network**
```
from sklearn.neural_network import MLPClassifier

cf = MLPClassifier()
param_grid = {"activation": ['identity', 'logistic', 'tanh', 'relu'], 
              "solver": ['lbfgs', 'sgd', 'adam'],
              "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
              "hidden_layer_sizes": [(50,50,50), (50,50), (100,100,100)]}
              
nn = MLPClassifier(activation='tanh', solver='adam', 
                   alpha=1e-05, hidden_layer_sizes=(100,100,100))
nn.fit(X_train_zi, y_train)
y_val_pred = nn.predict(X_val_zi)
metrics(y_val, y_val_pred)
```
**Logistic Regression**
```
from sklearn.linear_model import LogisticRegression

cf = LogisticRegression()
param_grid = {"solver": ['newton-cg','lbfgs', 'liblinear', 'sag'],
              "C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]}
              
lr = LogisticRegression(solver='lbfgs', C=1.0)
lr.fit(X_train_zi, y_train)
y_val_pred = lr.predict(X_val_zi)
metrics(y_val, y_val_pred)
```
**Stochastic Gradient Descent**
```
from sklearn.linear_model import SGDClassifier

cf = SGDClassifier()
param_grid = {"loss": ['hinge', 'log', 'modified_huber', 'perceptron'], 
              "penalty": ['none', 'l2', 'l1'],
              "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]}
 sgd = SGDClassifier(loss='perceptron', penalty='l2', alpha=0.01, n_iter=5, random_state=25)
sgd.fit(X_train_zi, y_train)
y_val_pred = sgd.predict(X_val_zi)
metrics(y_val, y_val_pred)
 ```

The Random Forest classifier seemed to perform best on the validation 
set. Therefore this classifier is tested with the testset. 
```
# Testing on testset 
X_train = np.concatenate((X_train, X_val), axis = 0)
y_train = np.concatenate((y_train, y_val), axis = 0)

inter = PolynomialFeatures(degree=2, interaction_only=True)
X_train_i = inter.fit_transform(X_train)
X_test_i = inter.transform(X_test)

model = RandomForestClassifier(bootstrap=False, min_samples_leaf=3, 
                            min_samples_split=2, criterion='gini', 
                            max_features=0.2, max_depth=None)
model.fit(X_train_i, y_train)
pred = model.predict(X_test_i)
metrics(y_test, pred)
```
Furthermore, I made ROC plots for all classifiers with the following code: 
```
probas = lr.fit(X_train, y_train).predict_proba(X_val)
fpr, tpr, thresholds = roc_curve(y_val, probas[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (sol, roc_auc))

plt.title('Receiver Operating Characteristic')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0, fontsize='small')
plt.show()
```
Finally, a big graph is created showing the ROC curves of all different classifiers: 
```
nb = GaussianNB()
nb.fit(X_train, y_train)
y_val_pred = nb.predict(X_val)

knn = KNeighborsClassifier(n_neighbors = 13)
knn.fit(X_train, y_train)
y_val_pred = knn.predict(X_val)

dt = DecisionTreeClassifier(criterion = 'entropy')
dt.fit(X_train, y_train)
y_val_pred = dt.predict(X_val)

rf = RandomForestClassifier(criterion = 'entropy')
rf.fit(X_train, y_train)
y_val_pred = rf.predict(X_val)

nn = MLPClassifier(activation = 'tanh', solver = 'adam')
nn.fit(X_train, y_train)
y_val_pred = nn.predict(X_val)

svc = SVC(kernel = 'poly')
svc.fit(X_train, y_train)
y_val_pred = svc.predict(X_val)

sgd = SGDClassifier(loss = 'log', random_state=25)
sgd.fit(X_train, y_train)
y_val_pred = sgd.predict(X_val)

lr = LogisticRegression(solver = 'newton-cg')
lr.fit(X_train, y_train)
y_val_pred = lr.predict(X_val)

# ROC
models = [
    {
        'label' : 'RandomForestClassifier',
        'model': rf,
        'roc_train': X_train,
        'roc_test': X_val,
        'roc_train_class': y_train,        
        'roc_test_class': y_val,        
    },   
    {
        'label' : 'SVC',
        'model': svc,
        'roc_train': X_train,
        'roc_test': X_val,
        'roc_train_class': y_train,        
        'roc_test_class': y_val,     
    },        
    {
        'label' : 'KNeighborsClassifier',
        'model': knn,
        'roc_train': X_train,
        'roc_test': X_val,
        'roc_train_class': y_train,        
        'roc_test_class': y_val,      
    },
    {
        'label' : 'LogisticRegression',
        'model': lr,
        'roc_train': X_train,
        'roc_test': X_val,
        'roc_train_class': y_train,        
        'roc_test_class': y_val,       
    },        
    {
        'label' : 'NaiveBayes',
        'model': nb,
        'roc_train': X_train,
        'roc_test': X_val,
        'roc_train_class': y_train,        
        'roc_test_class': y_val,       
    }, 
    {
        'label' : 'Neural Network',
        'model': nn,
        'roc_train': X_train,
        'roc_test': X_val,
        'roc_train_class': y_train,        
        'roc_test_class': y_val,       
    },
    {
        'label' : 'DecisionTree',
        'model': dt,
        'roc_train': X_train,
        'roc_test': X_val,
        'roc_train_class': y_train,        
        'roc_test_class': y_val,       
    }, 
    {
        'label' : 'StochasticGradientDescent',
        'model': sgd,
        'roc_train': X_train,
        'roc_test': X_val,
        'roc_train_class': y_train,        
        'roc_test_class': y_val,       
    } 
]


plt.clf()
plt.figure(figsize=(8,6))

for m in models:
    m['model'].probability = True
    probas = m['model'].fit(m['roc_train'], m['roc_train_class']).predict_proba(m['roc_test'])
    fpr, tpr, thresholds = roc_curve(m['roc_test_class'], probas[:, 1])
    roc_auc  = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], roc_auc))

plt.title('Receiver Operating Characteristic')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0, fontsize='small')
plt.show()
```

