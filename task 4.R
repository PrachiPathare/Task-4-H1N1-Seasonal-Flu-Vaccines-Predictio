options(warn = -1)
library(caret)
library(ggplot2)
library(glmnet)
library(dplyr)
library(randomForest)
library(readr)
library(DataExplorer)
library(tidyverse)
library(gmodels)
library(rsample)
library(class)
library(psych)
library(GGally)
library(pROC)

#reading the data
med_data = read.csv('H1N1_Flu_Vaccines.csv',row.names = 1, stringsAsFactors = FALSE)

summary(med_data)
str(med_data)

#cleaning the data

#removing the columns which have many NA values
med_data$health_insurance = NULL
med_data$employment_industry = NULL
med_data$employment_occupation = NULL

#removing NA values with the mode of the columns

getmode <- function(x) {
  uniqv <- unique(x)
  uniqv[which.max(tabulate(match(x, uniqv)))]
}
change_na = function(x){
  x[is.na(x)] = getmode(x)
  x
}
num_colms = c(1:21,33:34)
  #numeric_cols<- c(1:22, 33:34)
med_data[num_colms] = lapply(med_data[num_colms] , change_na)
med_data$household_adults[is.na(med_data$household_adults)] = getmode(med_data$household_adults)
med_data$household_children[is.na(med_data$household_children)] = getmode(med_data$household_children)


#factorizing char data
med_data$age_group = as.factor(med_data$age_group)
med_data$education = as.factor(med_data$education)
med_data$race= as.factor(med_data$race)
med_data$sex= as.factor(med_data$sex)
med_data$income_poverty= as.factor(med_data$income_poverty)
med_data$marital_status= as.factor(med_data$marital_status)
med_data$rent_or_own= as.factor(med_data$rent_or_own)
med_data$employment_status= as.factor(med_data$employment_status)
med_data$hhs_geo_region= as.factor(med_data$hhs_geo_region)
med_data$census_msa= as.factor(med_data$census_msa)



#checking the clean data

med_data %>% plot_missing()

#EDA

med_data %>% plot_intro()

#for continuous features

med_data %>% plot_histogram()

#categorial features
med_data %>% plot_bar()

#taking look at outliers
boxplot(med_data)


#H1N1 Vaccine taken
ggplot(med_data, aes(h1n1_vaccine,fill = h1n1_vaccine))+
  geom_bar() +
  labs(x= "h1n1 vaccines", y = "people taken") +
  guides(scale = "none")

#seasonal vaccine taken
ggplot(med_data, aes(seasonal_vaccine,fill = seasonal_vaccine))+
  geom_bar() +
  labs(x= "seasonal_vaccines", y = "people taken") +
  guides(scale = "none")


#EDA for h1n1 data and sesonal data
#studying bivariate plot

# h1n1 Vaccine taken by age group
ggplot(med_data, aes(h1n1_vaccine,fill = age_group))+
  geom_bar() +
  labs(x= "h1n1 vaccines", y = "people taken") +
  guides(scale = "none")
#more h1n1 vaccine is taken by 65+ age group and 18-34 years age group

# seasonal_vaccine taken by age group
ggplot(med_data, aes(seasonal_vaccine,fill = age_group))+
  geom_bar() +
  labs(x= "seasonal_vaccine", y = "people taken",title = "for sesonal vaccine") +
  guides(scale = "none")
#sesonal vaccine is mostly not taken by 65+
#seasonal vacine s mostly taken by 18-34 years


# education

#h1n1 vaccine taken by education
ggplot(med_data, aes(h1n1_vaccine,fill = education))+
  geom_bar() +
  labs(x= "h1n1 vaccines", y = "people taken") +
  guides(scale = "none")
#college students have taken more h1n1 vaccine 
#college graduate have taken least vaccine

# seasonal_vaccine taken by education
ggplot(med_data, aes(seasonal_vaccine,fill = education))+
  geom_bar() +
  labs(x= "seasonal_vaccine", y = "people taken",title = "for sesonal vaccine") +
  guides(scale = "none")
#college graduates are the one who have taken maximum seasonal vaccine 
#and they are the who have taken least vaccnie.

#income_poverty

#proportion of h1n1 vaccine by income_poverty
ggplot(med_data, aes(h1n1_vaccine,fill = income_poverty ))+
  geom_bar() +
  labs(x= "income_poverty", y = "h1n1 vaccine") +
  guides(scale = "none")
#income with <=75k$ have taken more vaccine

# seasonal_vaccine taken by income_poverty
ggplot(med_data, aes(seasonal_vaccine,fill = income_poverty))+
  geom_bar() +
  labs(x= "seasonal_vaccine", y = "people taken",title = "for sesonal vaccine") +
  guides(scale = "none")
#similar for seasonal

#sex

#proportion of h1n1 vaccine by sex
ggplot(med_data, aes(h1n1_vaccine,fill = sex ))+
  geom_bar() +
  labs(x= "values", y = "h1n1 vaccine") +
  guides(scale = "none")
#maximum vaccine have taken by females

# seasonal_vaccine taken by age group
ggplot(med_data, aes(seasonal_vaccine,fill = sex))+
  geom_bar() +
  labs(x= "seasonal_vaccine", y = "people taken",title = "for sesonal vaccine") +
  guides(scale = "none")
#seasonal vaccine proportion is fairly similar for male and female


#race

#proportion of h1n1 vaccine by race
ggplot(med_data, aes(h1n1_vaccine,fill = race ))+
  geom_bar() +
  labs(x= "values", y = "h1n1 vaccine") +
  guides(scale = "none")
#white race are the one who have taken more vaccine
#black race have taken least vaccine

#proportion of seasonal vaccine by race
ggplot(med_data, aes(seasonal_vaccine,fill = race))+
  geom_bar() +
  labs(x= "values", y = "seasonal_vaccine",title = "for sesonal vaccine") +
  guides(scale = "none")
#same is for sesonal vaccine



#finding correlations
med_num = med_data[-c(21:30)]
cor_data = round(cor(med_num),3)
corPlot(med_num)

med_data$h1n1_vaccine = as.factor(med_data$h1n1_vaccine)
med_data$seasonal_vaccine = as.factor(med_data$seasonal_vaccine)

#getting rid of the columns which are having negative or no correlation with target variables

data_h1n1 = med_data[-c(30,31,32)]

data_seas = med_data[-c(20,30,31,32)]



#partitioning data sets for h1n1

part_h1n1 = initial_split(data_h1n1, prop = 0.75, strata = 'h1n1_vaccine')
h1n1_train = training(part_h1n1)
h1n1_test = testing(part_h1n1)



#partitioning the dataset for seasonal vaccine

part_seas = initial_split(data_seas, prop = 0.75, strata = 'seasonal_vaccine')
seas_train = training(part_seas)
seas_test = testing(part_seas)
seas_train$household_adults[is.na(seas_train$household_adults)] = getmode(seas_train$household_adults)
seas_train$household_children[is.na(seas_train$household_children)] = getmode(seas_train$household_children)
seas_test$household_children[is.na(seas_test$household_children)] = getmode(seas_test$household_children)
seas_test$household_adults[is.na(seas_test$household_adults)] = getmode(seas_test$household_adults)

#Model building

mycontrol = trainControl(method = "cv", number = 5)


#desicion tree
set.seed(42)
d_t = train(h1n1_vaccine~., data = h1n1_train, method = "rpart", trControl = mycontrol, tuneLength = 5)
d_t

#prediction
pred_dt = predict(d_t , newdata = h1n1_test, type = "raw")

summary(pred_dt)


#confusion matrix

cm_dt = confusionMatrix(pred_dt, h1n1_test$h1n1_vaccine)
cm_dt
cm_dt$byClass[7]
cm_dt$byClass[6]
cm_dt$byClass[5]
df = data.frame(h1n1_test$h1n1_vaccine, pred_dt)
View(df)

#ROC curve
pred_dt_prob = predict(d_t, newdata = h1n1_test,type = "prob")
plot(roc(h1n1_test$h1n1_vaccine,pred_dt_prob[,2]))



#for seasonal vaccine
#Random forest
rf_seas = randomForest(seasonal_vaccine~., 
                       data = seas_train, 
                       mtry = 4, 
                       ntree = 2000, 
                       improtance = T, 
                       trControl = mycontrol)
rf_seas

pred_rf_seas = predict(rf_seas, newdata = seas_test,type ="response")
result_seas = data.frame(seas_test$h1n1_vaccine,pred_rf_seas)
View(result_seas)

#confusion matrix
cm_seas_rf = confusionMatrix(pred_rf_seas,seas_test$seasonal_vaccine)
cm_seas_rf
cm_seas_rf$byClass[7]
cm_seas_rf$byClass[6]
cm_seas_rf$byClass[5]

#roc curve
pred_rf_prob = predict(rf_seas,newdata = seas_test, type = "prob")
plot(roc(seas_test$seasonal_vaccine,pred_rf_prob[,2]))












