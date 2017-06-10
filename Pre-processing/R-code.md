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
library("lubridate")

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

In order to categeorize the activites per date, an extra variable is created,
indicating the activities per month, to see whether there are for example effects of season. 
Furthermore the variables are changed to numeric. 
```
# Create variable activity in month
data$Month <- month(data$activity_date)

# Changing strings with just numbers to numeric
data$languages <- as.numeric(data$languages)
data$Month <- as.numeric(data$Month)
```
For one feature the column name seemed unnecessary long 'Percepted Personality(foto/video/text)'.
therefore this name is shortened to just 'Personality'
```
names(data)[37] <- 'Personality'
```
While inspecting the summaries of the data, it showed that some values had spelling mistakes or different writing,
eventhough they clearly mean the same thing. For these value the writing is made consistent. 
```
data$category <- gsub("city tour", "City tour", fdata$category)
data$location <- gsub("Kula Lumpur", "Kuala Lumpur", fdata$location)
data$location <- gsub("Lisboa", "Lisbon", fdata$location)
data$location <- gsub("Roma", "Rome", fdata$location)
data$Personality <- gsub("artistic/exentric", "Artistic/exentric", 
       fdata$Personality)
data$Expertise <- gsub("photo/art", "Photo/art", fdata$Expertise)
data$Expertise <- gsub("history/architecture", "History/architecture", 
                        fdata$Expertise)
data$Expertise <- gsub("local lifestyle", "Local lifestyle", fdata$Expertise)
```
In order to do further analysis I converted the categorical variables to factors. 
```
for (x in c('location', 'category', 'host_continent', 
            'host_country', 'host_area', 'Character', 'Expertise',
            'Personality')) {
  data[,x] = factor(data[,x])
}  
```
Furthermore, categorical variable 'host_area' dummy variables are created. 
```
# Create dummy variables for cities ------------------------------------------
dummies <- model.matrix(~ + host_area,data)
data <- cbind(data, dummies)
```
Within the dataset some variables seem uniformative, because they only had one value for all instances,
because they had too many missing values (see Subsection 3.2.2 of thesis), or because they had 
unique values for every instance (like booking id's).
```
datamin <- data %>% select(-booking_id, -wine_tasting, -is_partner_booking, 
                           -email, -guest_country, -guest_gender, -fullname,
                           -has_published_experiences, -isOriginalsHost, 
                           -lang_sp, -lang_en, -age, -activity_id, -created,
                           -activity_date, -Character, -Personality, - Expertise,
                           -category, -est_age, -location, -host_continent,
                           -host_country, -host_area)
fdata <- as.data.frame(datamin)
```
Then, the instances with missing values are deleted from the data (again, see Subsection 3.2.2 of thesis). 
```
fdata <- na.omit(fdata)
```
Afther cleaning we investegate the dimensions of the data again. 
```
dim(fdata)
``` 
The dimensions now are:
- **Instances**: 38452
- **Features**: 53
