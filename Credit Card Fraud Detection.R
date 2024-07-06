knitr::opts_chunk$set(echo = TRUE, 
                      fig.align = 'center', 
                      cache = TRUE)


# Load all necessary libraries

if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(tidyr)) install.packages("tidyr")
if(!require(stringr)) install.packages("stringr")
if(!require(dplyr)) install.packages("dplyr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(gbm)) install.packages("gbm")
if(!require(caret)) install.packages("caret")
if(!require(xgboost)) install.packages("xgboost")
if(!require(e1071)) install.packages("e1071")
if(!require(class)) install.packages("class")
if(!require(ROCR)) install.packages("ROCR")
if(!require(randomForest)) install.packages("randomForest")
if(!require(PRROC)) install.packages("PRROC")
if(!require(reshape2)) install.packages("reshape2")
if(!require(lightgbm)) install.packages("lightgbm")

creditcard$Class2 <- ifelse(creditcard$Class == 0, "Legal", "Fraud")


# Create a histogram showing the proportion of genuine and fraudulent transactions

creditcard %>%
  ggplot(aes(Class2)) +
  geom_bar() +
  scale_x_discrete() +
  scale_y_continuous() +
  labs(title = "Proprotion of Fraudulent vs Genuine Transactions",
       x = "Type",
       y = "Frequency")


creditcard %>%
  select(Time, V1, V2, V3, V28, Amount, Class, Class2) %>%
  head(5)


# Construct a histogram of the amount of money being scammed by each fraudulent transaction

creditcard[creditcard$Class == 1,] %>%
  ggplot(aes(Amount)) + 
  geom_histogram(binwidth = 50) +
  labs(title = "Distribution of Fraud",
       x = "Dollars",
       y = "Frequency")


# Construct a table of the top 5 amounts of money being scammed

creditcard[creditcard$Class == 1,] %>%
  group_by(Amount) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  head(n=5)


#Construct a timeseries graph of all fraudulent transactions

creditcard[creditcard$Class == 1,] %>%
  ggplot(aes(Time)) + 
  geom_histogram(binwidth = 50) +
  labs(title = "Frauds Timeseries",
       x = "Time",
       y = "Frequency")


# Set seed

set.seed(284807)

# Split datasets

creditcard$Class2 <- NULL

train_index <- createDataPartition(
  y = creditcard$Class, 
  p = .6, 
  list = F
)

train <- creditcard[train_index,]

test_validation <- creditcard[-train_index,]

test_index <- createDataPartition(
  y = test_validation$Class, 
  p = .5, 
  list = F)

test <- test_validation[test_index,]
validation <- test_validation[-test_index,]

rm(train_index, test_index, test_validation)


# Set seed

set.seed(284807)

# Build baseline model that always predicts 'Genuine'

baseline <- creditcard
baseline$Class <- factor(0, c(0,1))

# Make predictions

pred <- prediction(
  as.numeric(as.character(baseline$Class)),
  as.numeric(as.character(creditcard$Class))
)

# Calculate the AUC

baseline_auc_value <- performance(pred, "auc")
baseline_auc_plot <- performance(pred, 'sens', 'spec')

# Construct plot

plot(baseline_auc_plot, main=paste("AUC:", baseline_auc_value@y.values[[1]]))

# Insert result into 'results' dataframe

results <- data.frame(
  Model = "Baseline", 
  AUC = baseline_auc_value@y.values[[1]]
)


# Set seed

set.seed(284807)

# Build Bayes model

bayes_model <- naiveBayes(Class ~ ., data = train, laplace = 1)

# Make predictions

predictions <- predict(bayes_model, newdata = test)
pred <- prediction(as.numeric(predictions), test$Class)

# Calculate the AUC

bayes_auc_value <- performance(pred, "auc")
bayes_auc_plot <- performance(pred, 'sens', 'spec')

# Construct plot

plot(bayes_auc_plot, main=paste("AUC:", bayes_auc_value@y.values[[1]]))

# Insert result into 'results' dataframe

results <- results %>% add_row(
  Model = "Bayes Model", 
  AUC = bayes_auc_value@y.values[[1]],
)


# Set seed

set.seed(284807)

# Build KNN Model

knn_model <- knn(train[,-30], test[,-30], train$Class, k=5, prob = TRUE)

#Make predictions

pred <- prediction(
  as.numeric(as.character(knn_model)),
  as.numeric(as.character(test$Class))
)

# Calculate the AUC

knn_auc_value <- performance(pred, "auc")
knn_auc_plot <- performance(pred, 'sens', 'spec')

# Construct plot

plot(knn_auc_plot, main=paste("AUC:", knn_auc_value@y.values[[1]]))

# Insert result into 'results' dataframe

results <- results %>% add_row(
  Model = "KNN Model", 
  AUC = knn_auc_value@y.values[[1]],
)


# Set seed

set.seed(284807) 

# Create training dataset

xgb_train <- xgb.DMatrix(
  as.matrix(train[, colnames(train) != "Class"]), 
  label = as.numeric(as.character(train$Class))
)

# Create test dataset

xgb_test <- xgb.DMatrix(
  as.matrix(test[, colnames(test) != "Class"]), 
  label = as.numeric(as.character(test$Class))
)

# Create validation dataset

xgb_validation <- xgb.DMatrix(
  as.matrix(validation[, colnames(validation) != "Class"]), 
  label = as.numeric(as.character(validation$Class))
)

# Prepare parameters

xgb_params <- list(
  objective = "binary:logistic", 
  eta = 0.1, 
  max.depth = 3, 
  nthread = 6, 
  eval_metric = "aucpr"
)

# Train XGBoost

xgb_model <- xgb.train(
  data = xgb_train, 
  params = xgb_params, 
  watchlist = list(test = xgb_test, validation = xgb_validation), 
  nrounds = 50, 
  early_stopping_rounds = 4,
  verbosity = 0,
  print_every_n = 10,
  silent = T,
  
)

# Make predictions

predictions <- predict(
  xgb_model, 
  newdata = as.matrix(test[, colnames(test) != "Class"]), 
  ntreelimit = xgb_model$bestInd
)
pred <- prediction(
  as.numeric(as.character(predictions)),
  as.numeric(as.character(test$Class))
)

# Calculate the AUC

xgb_auc_value <- performance(pred, "auc")
xgb_auc_plot <- performance(pred, 'sens', 'spec')

# Construct plot

plot(xgb_auc_plot, main=paste("AUC:", xgb_auc_value@y.values[[1]]))

# Insert result into 'results' dataframe

results <- results %>% add_row(
  Model = "XGBoost",
  AUC = xgb_auc_value@y.values[[1]])


# Show results

results