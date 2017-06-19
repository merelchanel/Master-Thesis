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
** X_train = (23070, 50) **
** X_val = (7691, 50) **
** X_test = (7691, 50) **

Applying PCA is tested to see whether it improves results. 
```
from sklearn.decomposition import PCA

# Apply regular PCA
pca = PCA()
X = pca.fit_transform(X)
```
However, it did not improve the results and therefore the results are not used. 

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
Then the different classifiers are tested on the validationset. 

**Naive Bayes**
```
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
y_val_pred = nb.predict(X_val)
report(y_val, y_val_pred)
```
**KNN**
```
from sklearn.neighbors import KNeighborsClassifier

for k in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_val_pred = knn.predict(X_val)
    print("K = ", k)
    report(y_val, y_val_pred)
``` 
**Decision Tree**
``` 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# tested with class_weights = {1:(1 till 10)}

for crit in ['entropy', 'gini']:
    dt = DecisionTreeClassifier(criterion = crit)
    dt.fit(X_train, y_train)
    y_val_pred = dt.predict(X_val)
    print("Criteria = ", crit)
    report(y_val, y_val_pred)
```
**Random Forest**
```
from sklearn.ensemble import RandomForestClassifier

# tested with class_weights = {1:(1 till 10)}

for crit in ['entropy', 'gini']:
    rf = RandomForestClassifier(criterion = crit)
    rf.fit(X_train, y_train)
    y_val_pred = rf.predict(X_val)
    print("crit = ", crit)
    report(y_val, y_val_pred)
```
**Neural Network**
```
from sklearn.neural_network import MLPClassifier

for act in ['identity', 'logistic', 'tanh', 'relu']:
    for sol in ['lbfgs', 'sgd', 'adam']:
        for a in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]:
            nn = MLPClassifier(activation = act, solver = sol, alpha = a)
            nn.fit(X_train, y_train)
            y_val_pred = nn.predict(X_val)
            print("activation = ", act, "solver = ", sol, "alpha =", a)
            report(y_val, y_val_pred)
```
**Support Vector Machine**
```
from sklearn.svm import SVC

for kern in ['rbf', 'poly', 'linear', 'sigmoid']:
    svc = SVC(kernel = kern)
    svc.fit(X_train, y_train)
    y_val_pred = svc.predict(X_val)
    print("kernel = ", kern)
    report(y_val, y_val_pred)
```
**Stochastic Gradient Descent**
```
from sklearn.linear_model import SGDClassifier

for loss in ['hing', 'log', 'modified_huber']:
        sgd = SGDClassifier(loss = loss, random_state=25)
        sgd.fit(X_train, y_train)
        y_val_pred = sgd.predict(X_val)
        print("Loss = ", loss)
        report(y_val, y_val_pred)
 ```
**Logistic Regression**
```
from sklearn.linear_model import LogisticRegression

for sol in ['newton-cg', 'lbfgs', 'liblinear', 'sag']:
    lr = LogisticRegression(solver = sol)
    lr.fit(X_train, y_train)
    y_val_pred = lr.predict(X_val)
    print("Solver = ", sol)
    report(y_val, y_val_pred)
```
The decision tree with the parameter setting criterion = 'entropy' seemed to perform best on the validation 
set. Therefore this classifier is tested with the testset. 
```
# Testing on testset 
X_train = np.concatenate((X_train, X_val), axis = 0)
y_train = np.concatenate((y_train, y_val), axis = 0)

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(X_train, y_train)
pred = model.predict(X_test)
report(y_test, pred)
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

