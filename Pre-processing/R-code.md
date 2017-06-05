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
To check the final dataset, we look at the dimensions and a summary of the data. 
```
dim(data)
summary(data)
``` 
The dimensions are:
- **Instances**: 39417
- **Features**: 67

A summary of the summary can be found in Appendix A of the thesis, where the average/range of the features is given. 

Then the data is checked for missing values and the number of unique values per feature. 
```
# Check missing values ---------------------------------------------------------
sapply(data,function(x) sum(is.na(x)))

# Check unique values ----------------------------------------------------------
sapply(data, function(x) length(unique(x)))
```
The number of missing and unique values can also be found in Appendix A of the thesis. 

Within the dataset some variables seem uniformative, because they only had one value for all instances,
or because they had too many missing values (see Subsection 3.2.2 of thesis). 
```
datamin <- data %>% select(-booking_id, -wine_tasting, -is_partner_booking, 
                           -email, -guest_country, -guest_gender, -fullname,
                           -has_published_experiences, -isOriginalsHost, 
                           -lang_sp, -lang_en, -age, -activity_id)
```

In order to do further analysis I converted the categorical variables to factors. 
```
fdata <- as.data.frame(datamin)

for (x in c('location', 'category', 'languages', 'host_continent', 
            'host_country', 'host_area', 'Character', 'Expertise',
            'Percepted Personality(foto/video/text)')) {
  fdata[,x] = factor(fdata[,x])
}  
```
Then, the instances with missing values are deleted from the data (again, see Subsection 3.2.2 of thesis). 
```
fdata <- na.omit(fdata)
```





