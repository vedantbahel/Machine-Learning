#setting up the directory
setwd("C:/Users/Student/Desktop/machinelearning_regression-master")

#loading data
data <- read.csv("House Data.csv")

#calling linear regression on called data
relation <- lm(price~sqft_living, data=data)
print(relation)

#predicting
a <- data.frame(sqft_living = 1044)
result <-  predict(relation,a)
print(result)

