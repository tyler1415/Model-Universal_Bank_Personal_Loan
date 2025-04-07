
# ---------------- USING SMOTE MODEL ---------------------

# Load libraries
library(tidyverse)
library(smotefamily) # for SMOTE
library(caret) # for confusion matrix


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

# -------------------- APPLY SMOTE -------------------- #

# Check class balance
table(train.df$PersonalLoan)

# Convert target to numeric 0/1 — required by smotefamily
train.df$PersonalLoan <- as.numeric(as.character(train.df$PersonalLoan))

# Separate features and target
train.features <- train.df[, setdiff(names(train.df), "PersonalLoan")]
train.target <- train.df$PersonalLoan

# Apply SMOTE (this version uses different naming)
smote.output <- SMOTE(X = train.features, target = train.target, K = 5)

# SMOTE returns:
# - $data: full dataset with synthetic samples
# - $syn_data: just synthetic samples
# - $orig_N and $syn_N: counts

# Rename target to match original
train.smote <- smote.output$data
colnames(train.smote)[ncol(train.smote)] <- "PersonalLoan"

# Make sure target is factor
train.smote$PersonalLoan <- as.factor(train.smote$PersonalLoan)

# Check new class balance
table(train.smote$PersonalLoan)

# -------------------- BUILD LOGISTIC MODEL -------------------- #
lrmodel.smote <- glm(PersonalLoan ~ ., data = train.smote, family = binomial)

# quick summary
summary(lrmodel.smote)

# calculate odds ratio
exp(coef(lrmodel.smote))

# -------------------- PREDICT ON TEST SET -------------------- #
lrmodel.smote.predictions <- predict(lrmodel.smote, newdata = test.df, type = "response")

# Convert to 0/1
lrmodel.smote.predictions <- ifelse(lrmodel.smote.predictions >= 0.5, 1, 0)

# -------------------- CONFUSION MATRIX -------------------- #
confusionMatrix(factor(lrmodel.smote.predictions), factor(test.df$PersonalLoan), positive = "1")