#' Calculate Mean Squared Error (MSE)
#'
#' @description 
#' Calculates the Mean Squared Error between actual and predicted continuous values.
#'
#' @param actual A numeric vector of actual values.
#' @param predicted A numeric vector of predicted values.
#'
#' @return A numeric value representing the MSE.
MSE <- function(actual, predicted) {
  mse <- mean((actual - predicted)^2)
  return(round(mse, 5))
}

#' Train Ratio Estimation Models (Linear Regression)
#'
#' @description 
#' Trains two multiple linear regression models to estimate the blending ratios 
#' of Taiwan and Vietnam rice, respectively.
#'
#' @param trainData The training dataset containing ONLY mixed rice samples.
#' @param features A vector of feature column names (default: X1 to X5).
#'
#' @return A list containing two trained `lm` objects: `modelT` (Taiwan ratio) and `modelV` (Vietnam ratio).
trainRatioModels <- function(trainData, features = c("X1", "X2", "X3", "X4", "X5")) {
  
  formulaT <- as.formula(paste("ratio_t ~", paste(features, collapse = " + ")))
  formulaV <- as.formula(paste("ratio_v ~", paste(features, collapse = " + ")))
  
  # Training Model
  lmModelT <- lm(formulaT, data = trainData)
  lmModelV <- lm(formulaV, data = trainData)
  
  return(list(
    modelT = lmModelT,
    modelV = lmModelV
  ))
}