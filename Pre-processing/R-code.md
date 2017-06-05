### Pre-processing

In this file the R-code used to pre-process the data is given. 

#### R-code
Load libraries and data.
```
library(readr)
library(dplyr)
library(ggplot2)
library(reshape2)
library('caret')
library(tidyr)

char <- 
  read_csv("C:/Users/Merel Tombrock/Desktop/WithLocals/Analysis/withlocals_originals_analysis/input/willems_golden host data_2016_12_29 - result - v_menno.csv")
age <- 
  read_csv("C:/Users/Merel Tombrock/Desktop/WithLocals/Analysis/withlocals_originals_analysis/input/age2.csv")
host <- 
  read_delim("C:/Users/Merel Tombrock/Desktop/WithLocals/Analysis/withlocals_originals_analysis/input/host-data-check.csv", 
                              ";", escape_double = FALSE, trim_ws = TRUE)
```
Then, the three datasets are joined together. 
```
data <- left_join(host, char, by = "email")
data <- left_join(data, age, by = "email")
```
