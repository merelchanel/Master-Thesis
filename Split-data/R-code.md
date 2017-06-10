### Split-data

In this file the R-code used to split the data is given. 

#### R-code
The data is splitted in two different ways in order to aid different analysis. Both of these splits are given. 

#### Split 1
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

#### Split 2
For the second split we are just interested in the data where chosen = 1, so in the chosen hosts. 

```
# Create subset of chosen ones
chosen <- subset(fdata, chosen == 1)
```
Then we split the data in characteristics of the activity and guests, which will serve as the input variables,
and the characteristics of the hosts, which will serve as prediction labels. 
```
vars <- names(chosen) %in% c("is_characteristic_outgoing", "is_characteristic_introvert", "is_personality_joie_de_vivre",
          "is_personality_knowledgeable", "is_personality_artistic", "is_expertise_unique_story",
          "is_expertise_food_drinks", "is_expertise_history", "is_expertise_local_lifestyle",
          "is_expertise_photo_art")
y <- chosen[vars]
chosen <- chosen[!vars]

datamin <- chosen %>% select(-host_area, -`(Intercept)`, -chosen)
X <- as.data.frame(datamin)
```
And again, we write these dataframes also to csv files. 
```
write.csv(X, file = "char_x.csv")
write.csv(y, file = "char_y.csv")
```
