library(InformationValue)
library(ROCR)
library(ggplot2)
library(vcdExtra)
library(DescTools)
library(tidyverse)
library(caret)
library(xgboost)
library(pdp)
library(stringr)

##############
# DATA SETUP #
##############

records <- read.csv("data/exercise01.csv")

str(records)

factor_cols <- c(
    "country", "workclass_name", "education_level", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex"
)

numeric_cols <- c(
    "age", "capital_gain", "capital_loss", "hours_week"
)

for (col in factor_cols) {
    records[, col] <- as.factor(records[, col])
}

# Make education_level an ordered factor
education_level_order <- c(
    "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th",
    "12th", "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors",
    "Masters", "Prof-school", "Doctorate"
)

records$education_level <- factor(
    records$education_level,
    order = TRUE,
    levels = education_level_order
)

# Make education_num an ordered factor
education_num_order <- c(
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
    "12", "13", "14", "15", "16"
)

records$education_num <- factor(
    records$education_num,
    order = TRUE,
    levels = education_num_order
)

set.seed(12345)

splits <- c(train = .7, validate = .2, test = .1)

groups <- sample(
    cut(
        seq(nrow(records)),
        nrow(records) * cumsum(c(0, splits)),
        labels = names(splits)
    )
)

split_results <- split(records, groups)

train <- split_results$train
validation <- split_results$validate
test <- split_results$test

#############################
# EXPLORATORY DATA ANALYSIS #
#############################

# Check for rare event
table(train$over_50k)
# No rare event

# Check for missing values
sapply(train, function(x) sum(is.na(x)))
# No NA's

# Check for abnormal values
for (col in numeric_cols) {
    print(col)
    print(summary(train[[col]]))
}

# Classifying values of 99999 for capital_gain as
# missing/unkown for the purpose of this model
# and imputing them with the median value
train$capital_gain_imputed <- ifelse(
    train$capital_gain == 99999,
    1,
    0
)
train$capital_gain[train$capital_gain == 99999] <-
    median(train$capital_gain, na.rm = TRUE)

# Check for quasi-complete separation which shouldn't
# be an issue for xgboost, but it's probably best to fix
# it anyway
for (col in factor_cols) {
    print(col)
    print(table(train[[col]], train$over_50k))
}

# Add new record to train to fix quasi-complete separation
# for workclass_name, but first create a new imputed
# flag column for each existing column
for (col in colnames(train)) {
    imputed_col_name <- paste(col, "imputed", sep = "_")
    if (!(col %in% c("id", "capital_gain_imputed", "over_50k"))) {
        train[[imputed_col_name]] <- 0
    }
}
train <- train %>%
    add_row(
        id = nrow(records) + 1,
        country = Mode(train$country),
        country_imputed = 1,
        age = round(median(train$age), digits = 0),
        age_imputed = 1,
        workclass_name = "Never-worked",
        workclass_name_imputed = 1,
        education_level = Mode(train$education_level),
        education_level_imputed = 1,
        education_num = Mode(train$education_num),
        education_num_imputed = 1,
        marital_status = Mode(train$marital_status),
        marital_status_imputed = 1,
        occupation = Mode(train$occupation),
        occupation_imputed = 1,
        relationship = Mode(train$relationship),
        relationship_imputed = 1,
        race = Mode(train$race),
        race_imputed = 1,
        sex = Mode(train$sex),
        sex_imputed = 1,
        capital_gain = round(median(train$capital_gain), digits = 0),
        capital_gain_imputed = 1,
        capital_loss = round(median(train$capital_loss), digits = 0),
        capital_loss_imputed = 1,
        hours_week = round(median(train$hours_week), digits = 0),
        hours_week_imputed = 1,
        over_50k = 1
    )

# Need to make sure the columns match up
for (col in colnames(validation)) {
    imputed_col_name <- paste(col, "imputed", sep = "_")
    if (!(col %in% c("id", "over_50k"))) {
        validation[[imputed_col_name]] <- 0
    }
}

# Grab 2 record from the validation set to fix the lack
# of observations for Holand-Netherlands in the country column
matching_non_events <-
    validation[
        validation$country == "Holand-Netherlands" & validation$over_50k == 0
    , ]
matching_events <-
    validation[
        validation$country == "Holand-Netherlands" & validation$over_50k == 1
    , ]

# Make sure we actually have at least one of each
nrow(matching_non_events)
nrow(matching_events)
# Only have 1 for matching_non_events and none for matching_events

# Add the one observation to train and drop from validation
train <- rbind(train, matching_non_events[1, ])
validation <-
    validation[validation$id != matching_non_events[1, ]$id, ]

train <- train %>%
    add_row(
        id = nrow(records) + 1,
        country = "Holand-Netherlands",
        country_imputed = 1,
        age = round(median(train$age), digits = 0),
        age_imputed = 1,
        workclass_name = Mode(train$workclass_name),
        workclass_name_imputed = 1,
        education_level = Mode(train$education_level),
        education_level_imputed = 1,
        education_num = Mode(train$education_num),
        education_num_imputed = 1,
        marital_status = Mode(train$marital_status),
        marital_status_imputed = 1,
        occupation = Mode(train$occupation),
        occupation_imputed = 1,
        relationship = Mode(train$relationship),
        relationship_imputed = 1,
        race = Mode(train$race),
        race_imputed = 1,
        sex = Mode(train$sex),
        sex_imputed = 1,
        capital_gain = round(median(train$capital_gain), digits = 0),
        capital_gain_imputed = 1,
        capital_loss = round(median(train$capital_loss), digits = 0),
        capital_loss_imputed = 1,
        hours_week = round(median(train$hours_week), digits = 0),
        hours_week_imputed = 1,
        over_50k = 1
    )

# convert character columns and imputed flag columns to factors
for (col in colnames(train)) {
    if (grepl("imputed", col, fixed = TRUE)) {
        train[[col]] <- factor(train[[col]], levels = c("0", "1"))
    } else if (typeof(train[[col]]) == "character") {
        train[[col]] <- as.factor(train[[col]])
    }
}

for (col in colnames(validation)) {
    if (grepl("imputed", col, fixed = TRUE)) {
        validation[[col]] <- factor(validation[[col]], levels = c("0", "1"))
    } else if (typeof(validation[[col]]) == "character") {
        validation[[col]] <- as.factor(validation[[col]])
    }
}

# Sanity check to make sure we fixed the quasi-complete separation
for (col in factor_cols) {
    print(col)
    print(table(train[[col]], train$over_50k))
}

# Dropping education_num due to perfect multicollinearity
# with education_level
train <- subset(train, select = -c(education_num, education_num_imputed))

#########################
# INITIAL XGBOOST MODEL #
#########################

set.seed(12345)

train_x <- model.matrix(over_50k ~ . - id, data = train)[, -1]
train_y <- train$over_50k

xgboost_train <- xgb.DMatrix(data = train_x, label = train_y)

# 10 fold cross validation model testing the root mean squared error
# for 200 different rounds
xgboost_model <- xgb.cv(
    data = xgboost_train,
    nrounds = 200,
    objective = "binary:logistic",
    nfold = 10
)

# Which number of rounds has the lowest test rmse
n_rounds <- which(
    xgboost_model$evaluation_log$test_logloss_mean ==
        min(xgboost_model$evaluation_log$test_logloss_mean)
)

# Tuning through caret initialization
tune_grid <- expand.grid(
    nrounds = n_rounds,  # 61
    eta = c(0.1, 0.15, 0.2, 0.25, 0.3),
    max_depth = c(1:10),
    gamma = c(0),
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = c(0.25, 0.5, 0.75, 1)
)

train_y <- as.factor(train_y)

set.seed(12345)

# Only do this the first time before the best model
# is found since it's computationally intensive (took ~16 hours).
# Find the model that is tuned with the best parameters
# that were chosen above to test (1 to 10 different depth
# levels, 5 potential eta values and 4 potential subsample values)

# using 10-fold cross-validation
xgboost_caret <- train(
    x = train_x,
    y = train_y,
    method = "xgbTree",
    tuneGrid = tune_grid,
    trControl = trainControl(method = "cv", number = 10)
)

# Look for the lowest point on this graph
plot(xgboost_caret)

# Inputting the eta, max_depth, subsample, and nrounds that minimized
# the log loss in the caret cross validation model
eta <- xgboost_caret$bestTune$eta
max_depth <- xgboost_caret$bestTune$max_depth
subsample <- xgboost_caret$bestTune$subsample
n_rounds <- xgboost_caret$bestTune$nrounds

# Comment this section out if you don't want to tune the model again
# These values are the same as the parameters above, just manually set
# eta <- 0.25
# max_depth <- 6
# subsample <- 1
# n_rounds <- 61

set.seed(12345)
train_y <- train$over_50k

best_xgboost <- xgboost(
    data = train_x,
    label = train_y,
    eta = eta,
    nrounds = n_rounds,
    max_depth = max_depth,
    subsample = subsample,
    objective = "binary:logistic"
)

# Checking variable importance
xgboost_important_vars <- xgb.importance(
    feature_names = colnames(train_x),
    model = best_xgboost
)

xgb.ggplot.importance(xgboost_important_vars)

xgboost_important_vars
#                                 Feature         Gain        Cover    Frequency   Importance
#  1:    marital_statusMarried-civ-spouse 3.517263e-01 0.0556354156 0.0234063745 3.517263e-01
#  2:                        capital_gain 1.786322e-01 0.1779072124 0.1120517928 1.786322e-01
#  3:                   education_level.L 1.637639e-01 0.0707200958 0.0408366534 1.637639e-01
#  4:                                 age 7.940966e-02 0.1145300563 0.1882470120 7.940966e-02
#  5:                        capital_loss 6.669363e-02 0.1069587089 0.0861553785 6.669363e-02
#  6:                          hours_week 4.123932e-02 0.0691817887 0.1085657371 4.123932e-02
#  7:           occupationExec-managerial 1.530486e-02 0.0244879311 0.0169322709 1.530486e-02
#  8:            occupationProf-specialty 9.603331e-03 0.0173226913 0.0169322709 9.603331e-03
#  9:             occupationOther-service 8.559204e-03 0.0185820146 0.0074701195 8.559204e-03
# 10:                    relationshipWife 7.394748e-03 0.0132760764 0.0214143426 7.394748e-03

# Check how well variables do against random variable
set.seed(12345)
train$random <- rnorm(nrow(train))
train_x_random <- model.matrix(over_50k ~ . - id, data = train)[, -1]

best_xgboost_random <- xgboost(
    data = train_x_random,
    label = train_y,
    eta = eta,
    nrounds = n_rounds,
    max_depth = max_depth,
    subsample = subsample,
    objective = "binary:logistic"
)

xgboost_important_vars_random <- xgb.importance(
    feature_names = colnames(train_x_random),
    model = best_xgboost_random
)

xgb.ggplot.importance(xgboost_important_vars_random)

# Random variable "outperforms" all other variables except
# marital_status, capital_gain, education_level, age,
# capital_loss, and hours_week

# Drop all variables that did worse than the random variable
cols_to_keep <- c(
    "id",
    "marital_status",
    "capital_gain",
    "education_level",
    "age",
    "capital_loss",
    "hours_week",
    "over_50k"
)
train <- subset(train, select = cols_to_keep)

set.seed(12345)
train_x <- model.matrix(over_50k ~ . - id, data = train)[, -1]
train_y <- train$over_50k

best_xgboost <- xgboost(
    data = train_x,
    label = train_y,
    eta = eta,
    nrounds = n_rounds,
    max_depth = max_depth,
    subsample = subsample,
    objective = "binary:logistic"
)

###############################
# INITIAL MODEL ON VALIDATION #
###############################

# Imputing the same way for validation as we did
# for training
validation$capital_gain[validation$capital_gain == 99999] <-
    median(validation$capital_gain, na.rm = TRUE)

validation <- subset(validation, select = cols_to_keep)

validation_x <- model.matrix(over_50k ~ . - id, data = validation)[, -1]
validation_y <- validation$over_50k

# Calculate the area under the ROC curve on validation data
xgboost_validation_predictions <- predict(
    best_xgboost,
    type = "prob",
    newdata = validation_x
)

validation_concordance <- Concordance(
    validation_y,
    xgboost_validation_predictions
)
validation_concordance
# $Concordance
# [1] 0.9220834
#
# $Discordance
# [1] 0.07791661
#
# $Tied
# [1] 0
#
# $Pairs
# [1] 17153416

validation_predictions <- ROCR::prediction(
    xgboost_validation_predictions,
    validation_y
)
validation_performance <- performance(
    validation_predictions,
    measure = "tpr",
    x.measure = "fpr"
)
validation_auroc <-
    validation_concordance$Concordance + (0.5 * validation_concordance$Tied)
validation_auroc
# 0.9220834


# Create df with true and false positive rate at different cutoffs
performance_measures <- data.frame(
    validation_performance@x.values[[1]],
    validation_performance@y.values[[1]]
)

# Adjust names
names(performance_measures) <- str_to_title(
    c(validation_performance@x.name, validation_performance@y.name)
)

# Create ROC curve plot for XGBoost on validation set
performance_measures %>%
    ggplot(aes(x = `False Positive Rate`, y = `True Positive Rate`)) +
    geom_line() +
    geom_ribbon(ymin = 0, aes(ymax = `True Positive Rate`), fill = "darkblue") +
    annotate(
        "text",
        x = 0.50,
        y = 0.50,
        label = paste0(
            "AUROC = ",
            format(round(validation_auroc, 3), nsmall = 3)
        ),
        color = "white",
        size = 10
    ) +
    labs(
        title = "XGBoost ROC Curve",
        subtitle = "On The Validation Data"
    ) +
    theme_bw() +
    theme(
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5)
    )

###################
# FINE TUNE MODEL #
###################

combined <- rbind(train, validation)

set.seed(12345)

combined_x <- model.matrix(over_50k ~ . - id, data = combined)[, -1]
combined_y <- combined$over_50k

# Replacing "-", "^", and "." with "_" since the pdp plots hate
# those special characters
colnames(combined_x) <- str_replace_all(
    colnames(combined_x),
    c("-" = "_", "\\^" = "_", "\\." = "_")
)

xgboost_combined <- xgb.DMatrix(data = combined_x, label = combined_y)

# 10 fold cross validation model testing the root mean squared error
# for 200 different rounds
xgboost_combined_model <- xgb.cv(
    data = xgboost_combined,
    nrounds = 200,
    objective = "binary:logistic",
    nfold = 10
)

# Which number of rounds has the lowest test rmse
n_rounds <- which(
    xgboost_combined_model$evaluation_log$test_logloss_mean ==
        min(xgboost_combined_model$evaluation_log$test_logloss_mean)
)

# Retuning through caret initialization
# Using smaller windows based on training parameters
tune_grid <- expand.grid(
    nrounds = n_rounds,  # 47
    eta = c(0.2, 0.25, 0.3),
    max_depth = c(4:8),
    gamma = c(0),
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = c(0.75, 1)
)

combined_y <- as.factor(combined_y)

set.seed(12345)

# Only do this the first time before the best model
# is found since it's computationally intensive.
# Find the model that is tuned with the best parameters
# that were chosen above to test (4 to 8 different depth
# levels, 3 potential eta values and 2 potential subsample values)

# Using 10-fold cross-validation
combine_xgboost_caret <- train(
    x = combined_x,
    y = combined_y,
    method = "xgbTree",
    tuneGrid = tune_grid,
    trControl = trainControl(method = "cv", number = 10)
)

# Look for the lowest point on this graph
plot(combine_xgboost_caret)

# Inputting the eta, max_depth, subsample, and nrounds that minimized
# the log loss in the caret cross validation model
eta <- combine_xgboost_caret$bestTune$eta
max_depth <- combine_xgboost_caret$bestTune$max_depth
subsample <- combine_xgboost_caret$bestTune$subsample
n_rounds <- combine_xgboost_caret$bestTune$nrounds

# Comment this section out if you don't want to tune the model again
# These values are the same as the parameters above, just manually set
# eta <- 0.25
# max_depth <- 6
# subsample <- 1
# n_rounds <- 47

set.seed(12345)
combined_y <- combined$over_50k

final_xgboost <- xgboost(
    data = combined_x,
    label = combined_y,
    eta = eta,
    nrounds = n_rounds,
    max_depth = max_depth,
    subsample = subsample,
    objective = "binary:logistic"
)

xgboost_important_vars <- xgb.importance(
    feature_names = colnames(combined_x),
    model = final_xgboost
)

# Variable importance
xgb.ggplot.importance(xgboost_important_vars)

xgboost_important_vars
#                                 Feature         Gain        Cover    Frequency   Importance
#  1:    marital_statusMarried_civ_spouse 3.937302e-01 8.956220e-02 0.0400500626 3.937302e-01
#  2:                        capital_gain 1.983051e-01 2.245304e-01 0.1877346683 1.983051e-01
#  3:                   education_level_L 1.920860e-01 1.502059e-01 0.0782227785 1.920860e-01
#  4:                                 age 8.476785e-02 1.811068e-01 0.2428035044 8.476785e-02
#  5:                        capital_loss 7.622161e-02 1.767533e-01 0.1483103880 7.622161e-02
#  6:                          hours_week 4.422525e-02 1.276793e-01 0.1702127660 4.422525e-02
#  7:     marital_statusMarried_AF_spouse 1.853900e-03 2.170386e-02 0.0087609512 1.853900e-03
#  8:         marital_statusNever_married 1.512752e-03 6.680495e-03 0.0100125156 1.512752e-03
#  9:                  education_level_14 1.510962e-03 1.597322e-03 0.0081351690 1.510962e-03
# 10:                  education_level_13 8.752524e-04 2.208556e-03 0.0168961202 8.752524e-04

################
# TEST METRICS #
################

# Imputing the same way for test as we did
# for training
test$capital_gain[test$capital_gain == 99999] <-
    median(test$capital_gain, na.rm = TRUE)

test <- subset(test, select = cols_to_keep)

test_x <- model.matrix(over_50k ~ . - id, data = test)[, -1]
test_y <- test$over_50k

# Calculate the area under the ROC curve on test data
xgboost_test_predictions <- predict(
    final_xgboost,
    type = "prob",
    newdata = test_x
)

test_concordance <- Concordance(
    test_y,
    xgboost_test_predictions
)

test_concordance
# $Concordance
# [1] 0.9132666
#
# $Discordance
# [1] 0.08673342
#
# $Tied
# [1] 0
#
# $Pairs
# [1] 4269300

test_predictions <- ROCR::prediction(
    xgboost_test_predictions,
    test_y
)
test_performance <- performance(
    test_predictions,
    measure = "tpr",
    x.measure = "fpr"
)
test_auroc <-
    test_concordance$Concordance + (0.5 * test_concordance$Tied)

test_auroc
# 0.9132666

###############
# FINAL PLOTS #
###############

combined_x <- model.matrix(over_50k ~ . - id, data = combined)[, -1]
combined_y <- combined$over_50k

# Replacing "-", "^", and "." with "_" since the pdp plots hate
# those special characters
colnames(combined_x) <- str_replace_all(
    colnames(combined_x),
    c("-" = "_", "\\^" = "_", "\\." = "_")
)

######### ROC CURVE #########

# Create df with true and false positive rate at different cutoffs
performance_measures <- data.frame(
    test_performance@x.values[[1]],
    test_performance@y.values[[1]]
)

# Adjust names
names(performance_measures) <- str_to_title(
    c(test_performance@x.name, test_performance@y.name)
)

# Create ROC curve plot for XGBoost on test set
performance_measures %>%
    ggplot(aes(x = `False Positive Rate`, y = `True Positive Rate`)) +
    geom_line() +
    geom_ribbon(ymin = 0, aes(ymax = `True Positive Rate`), fill = "darkblue") +
    annotate(
        "text",
        x = 0.50,
        y = 0.50,
        label = paste0(
            "AUROC = ",
            format(round(test_auroc, 3), nsmall = 3)
        ),
        color = "white",
        size = 10
    ) +
    labs(
        title = "XGBoost ROC Curve",
        subtitle = "On The Test Data"
    ) +
    theme_bw() +
    theme(
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5)
    )

######### PARTIAL DEPENDENCY PLOTS #########

gains_pdp_plot <- pdp::partial(
    final_xgboost, pred.var = "capital_gain",
    plot = FALSE, alpha = 1, prob = TRUE,
    type = "classification", plot.engine = "ggplot2",
    train = combined_x, pdp.color = "red"
)

marital_pdp_plot <- pdp::partial(
    final_xgboost, pred.var = "marital_statusMarried_civ_spouse",
    plot = FALSE, alpha = 1, prob = TRUE,
    type = "classification", plot.engine = "ggplot2",
    train = combined_x, pdp.color = "red"
)

ggplot() +
    geom_line(
        data = gains_pdp_plot,
        aes(x = capital_gain, y = yhat, color = "Capital Gain")
    ) +
    labs(
        x = "Capital Gain",
        y = "Predicted Probability of Earning > $50k Per Year",
        title = "Partial Dependency Plot of Capital Gain"
    ) +
    theme_bw() +
    theme(
        legend.position = "none",
        plot.title = element_text(hjust = 0.5)
    )

ggplot() +
    geom_line(
        data = marital_pdp_plot,
        aes(
            x = marital_statusMarried_civ_spouse,
            y = yhat,
            color = "Marital Status"
        )
    ) +
    labs(
        x = "Married to Civilian",
        y = "Predicted Probability of Earning > $50k Per Year",
        title = "Partial Dependency Plot of Being Married to a Civilian"
    ) +
    theme_bw() +
    theme(
        legend.position = "none",
        plot.title = element_text(hjust = 0.5)
    )

# EVERYTHING BELOW THIS LINE DOES NOT NEED TO BE RUN

set.seed(12345)
# Create the predictor but make it predict using an xgb.DMatrix
xgboost_combined_predictions <- Predictor$new(
    final_xgboost,
    data = as.data.frame(combined_x),
    y = combined_y,
    predict.func = function(model, newdata) {
        print(colnames(newdata))
        new_data_x <- xgb.DMatrix(data.matrix(newdata))
        results <- predict(model, new_data_x)
        return(results)
    }
)

# I honestly have no idea what's going on here.
# as.data.frame(combined_x) has the correct amount
# of columns (25), but when it gets passed in the
# predict.fun function on line 721, it only has 9.
#
# The categorical columns that ARE present in newdata
# are properly one-hot encoded (e.g. marital_statusSeparated,
# marital_statusWidowed, etc.), but it just doesn't have all
# of them.
#
# Because of that trying to run the code block starting on line 745
# results in the following error:
# Error in predict.xgb.Booster(model, new_data_x) :
#   Feature names stored in `object` and `newdata` are different!

# Partial dependency plot plot for capital gain
pdp_plot <- FeatureEffects$new(
    xgboost_combined_predictions,
    method = "pdp",
    features = "capital_gain"
)
