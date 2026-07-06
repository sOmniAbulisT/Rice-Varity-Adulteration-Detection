# ==============================================================================
# Two-Stage Mixed Rice Prediction Pipeline
# ==============================================================================

# 1. Load required packages and custom modules
library(caret)
library(e1071)
library(randomForest)
library(glmnet)

source("src/01_data_cleaning.R")
source("src/02_classification.R")
source("src/03_ratio_estimation.R")

# ==============================================================================
# Core Business Logic: Two-Stage Prediction (MRPM)
# ==============================================================================

#' Mixed Rice Prediction Model (MRPM) Pipeline
#'
#' @description 
#' Executes a two-stage prediction: first using a classifier to determine the origin. 
#' If classified as "Mixed", it triggers the second-stage linear regression models 
#' to estimate the blending ratios of Taiwan and Vietnam rice, along with 95% 
#' confidence intervals.
#'
#' @param newData The new dataset for prediction (data.frame).
#' @param classifier The trained classification model (e.g., SVM).
#' @param ratioModelT The trained linear model for estimating the Taiwan rice ratio.
#' @param ratioModelV The trained linear model for estimating the Vietnam rice ratio.
#'
#' @return A data.frame containing the predicted classes and estimated ratios with CIs.
MRPM <- function(newData, classifier, ratioModelT, ratioModelV) {
  
  n_rows <- nrow(newData)
  
  # Technique 1: Pre-allocation for memory efficiency
  # Initialize the full-sized dataframe upfront to avoid slow rbind operations in loops
  resultTable <- data.frame(
    Sample_ID = rownames(newData),
    Actual_Class = if("Country" %in% colnames(newData)) newData$Country else NA,
    Predict_Class = rep(NA, n_rows),
    Ratio_Taiwan = rep(NA, n_rows),
    CI_95_Taiwan = rep(NA, n_rows),
    Ratio_Vietnam = rep(NA, n_rows),
    CI_95_Vietnam = rep(NA, n_rows),
    stringsAsFactors = FALSE
  )
  
  # ----------------------------------------------------
  # Stage 1: Origin Classification
  # ----------------------------------------------------
  classPreds <- predict(classifier, newdata = newData)
  resultTable$Predict_Class <- classPreds
  
  # ----------------------------------------------------
  # Stage 2: Ratio Estimation for Mixed Rice
  # ----------------------------------------------------
  # Technique 2: Vectorized Operations
  # Extract indices of all "Mixed" rows and process them simultaneously
  mixedIndices <- which(classPreds == "Mixed")
  
  if (length(mixedIndices) > 0) {
    mixedData <- newData[mixedIndices, ]
    
    # Predict and obtain 95% confidence intervals
    predT <- predict(ratioModelT, newdata = mixedData, interval = "confidence", level = 0.95)
    predV <- predict(ratioModelV, newdata = mixedData, interval = "confidence", level = 0.95)
    
    # Assign fitted values and confidence bounds to the pre-allocated table
    resultTable$Ratio_Taiwan[mixedIndices] <- round(predT[, "fit"], 3)
    resultTable$CI_95_Taiwan[mixedIndices] <- paste0("[", round(predT[, "lwr"], 3), ", ", round(predT[, "upr"], 3), "]")
    
    resultTable$Ratio_Vietnam[mixedIndices] <- round(predV[, "fit"], 3)
    resultTable$CI_95_Vietnam[mixedIndices] <- paste0("[", round(predV[, "lwr"], 3), ", ", round(predV[, "upr"], 3), "]")
  }
  
  return(resultTable)
}


# ==============================================================================
# Execution Script
# ==============================================================================
# Note: Reviewers or users can run this block to see the end-to-end result.

if (FALSE) { # Wrapped in if (FALSE) to prevent auto-execution when sourcing the script
  
  # Set seed for reproducibility
  set.seed(2026)
  
  # Step 1: Read and prepare data
  raw_data <- read.csv("data/2024SL_train_data.csv")
  taiwan_data <- raw_data[raw_data$Country == "Taiwan", ]
  vietnam_data <- raw_data[raw_data$Country == "Vietnam", ]
  
  # Generate 1000 mixed rice samples
  m2 <- generateMixedSample(data = raw_data, times = 1000) 
  clean_dataset <- rbind(
    cbind(taiwan_data[, 2:7], ratio_t = 1, ratio_v = 0, Country = "Taiwan"),
    cbind(vietnam_data[, 2:7], ratio_t = 0, ratio_v = 1, Country = "Vietnam"),
    m2
  )
  
  # Split into training and testing sets
  split_data <- prepareTrainingSet(clean_dataset, trainRatio = 0.8)
  train_data <- split_data$train
  test_data <- split_data$test
  
  # Step 2: Train the stage 1 classification model (using SVM Polynomial as an example)
  my_svm <- trainSvmClassifier(train_data, kernelType = "polynomial")
  
  # Step 3: Extract mixed rice data from the training set to train the stage 2 regression models
  mixed_train <- train_data[train_data$Country == "Mixed", ]
  my_ratio_models <- trainRatioModels(mixed_train)
  
  # Step 4: MRPM
  final_report <- MRPM(
    newData = test_data,
    classifier = my_svm,
    ratioModelT = my_ratio_models$modelT,
    ratioModelV = my_ratio_models$modelV
  )
  
  # Step 5: Output the results
  head(final_report)
  write.csv(final_report, "prediction_results.csv", row.names = FALSE)
}