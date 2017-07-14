### Exploratory Data Analysis (EDA)

While pre-processing the data also some EDA was performed to investiage the data. 

To investigate the strenght of the associations between nominal values, Cramer's V is calculated. 
``` 
cv.test = function(x,y) {
CV = sqrt(chisq.test(x, y, correct=FALSE)$statistic /
(length(x) * (min(length(unique(x)),length(unique(y))) - 1)))
print.noquote("Cramér V / Phi:")
return(as.numeric(CV))
}

cv.test(fdata$location, fdata$category) # Repeat with different features.
```
The results of Cramer's V can be found in Subsection 3.2.2 of the thesis. 

### Barplots
To visually investigate the distribution of choices.
```
ggplot(data, aes(x = chosen)) + geom_bar()
```
To investigate the character of the hosts between the choices. 
```
ggplot(fdata, aes(chosen, fill=Character)) + geom_bar()
``` 
To investigate the expertises between the chacarter of the hosts. 
```
ggplot(fdata, aes(Character, fill=Expertise)) + geom_bar(position="dodge")
```
An example of a stacked histogram that was created is found below: 
![stackedhistogram](/images/hist.png)
Format: ![Alt Text](url)
