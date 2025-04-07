
# ---------------- NO SMOTE MODEL ---------------------
# Load libraries
library(tidyverse)
library(caret)

# Load data
bank.df <- read.csv("C:\\Users\\tgmce\\Downloads\\Universal Bank Dataset.csv")

# view data structures
str(bank.df)

# summary of data and visualization of columns
summary(bank.df)

ggplot(gather(bank.df), aes(value)) +
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free') +
  theme_bw()

# get number of rows in data frame
print(nrow(bank.df))

# set a seed to ensure computational reproducibility
set.seed(1)

# get indexes from 1 to 5000
indx <- sample(nrow(bank.df), nrow(bank.df) * 0.75)

# select the data with the indexes sampled
train.df <- bank.df[indx, ]
# select the data with the indexes not sampled
test.df <- bank.df[-indx, ]

# build the logistic regression model
lrmodel1 <- glm(PersonalLoan~., data = train.df, family=binomial)

# quick summary
summary(lrmodel1)

# calculate odds ratio by exponentiation the coefficients
exp(coef(lrmodel1))

# logistic regression predictions
lrmodel1.predictions <- predict(lrmodel1, newdata = test.df, type = "response")
# convert predictions to 0/1
lrmodel1.predictions <- ifelse(lrmodel1.predictions >= 0.5, 1, 0)
# view predictions
lrmodel1.predictions

# view predictions
tab <- table(test.df$PersonalLoan, lrmodel1.predictions)
confusionMatrix(factor(lrmodel1.predictions), factor(test.df$PersonalLoan), positive = "1")

# compute accuracy: total correct/ total observations
sum(diag(tab))/sum(tab)