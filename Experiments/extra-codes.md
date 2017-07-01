### Code used for the extra experiments in Phyton. 

```
### Extract feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print('Feature Ranking:')
for f in range(X_train_i.shape[1]):
    if importances[indices[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        #print ("feature name: ", chosen[indices[f]])
```
