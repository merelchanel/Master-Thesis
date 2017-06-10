### Split-data

In this file the R-code used to split the data is given. 

#### R-code
The data is splitted in two different ways in order to aid different analysis. Both of these splits are given. 

The first way is: use the data as it was after the pre-processing, 
so the target label 'Chosen' is seperated from the prediction variables. 
```
# Split the y variable 
y <- fdata['chosen']
X <- fdata
X$chosen <- NULL
```
Then, export the dataframes to csv files, to be used in Phyton. 
```
write.csv(X, file = "chosen_x.csv")
write.csv(y, file = "chosen_y.csv")
```
