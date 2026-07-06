library(caret)
library(randomForest)
library(e1071)
library(glmnet)
library(MASS)
library(pls)

#' Calculate Confusion Matrix Accuracy
#'
#' @description 
#' Calculates the overall accuracy from a given confusion matrix.
#'
#' @param conf_matrix A table representing the confusion matrix.
#' @return A numeric value representing the accuracy (0 to 1).
cfmAccuracy <- function(conf_matrix) {
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  return(round(accuracy, 4))
}

#' Train Linear Discriminant Analysis (LDA) Classifier
#'
#' @description 
#' Trains an LDA model using 10-fold cross-validation via the `caret` package.
#'
#' @param trainData The training dataset (data.frame).
#' @param features A vector of feature column names (default: X1 to X5).
#' @param target The target column name (default: "Country").
#' @param folds Number of cross-validation folds (default: 10).
#'
#' @return A trained caret LDA model object.
trainLdaClassifier <- function(trainData, features = c("X1", "X2", "X3", "X4", "X5"), target = "Country", folds = 10) {
  
  # Country ~ X1 + X2 + X3 + X4 + X5
  formula <- as.formula(paste(target, "~", paste(features, collapse = " + ")))
  
  # Setting the parameter for CV
  cvControl <- trainControl(method = "cv", number = folds)
  
  # Training model
  ldaModel <- train(
    formula, 
    data = trainData, 
    method = "lda", 
    trControl = cvControl
  )
  
  return(ldaModel)
}


#' Train Partial Least Squares (PLS) Classifier
#'
#' @description 
#' Adapts the `plsr` function for classification by automatically generating dummy 
#' variables (one-hot encoding) for the target class. 
#'
#' @param trainData The training dataset (data.frame).
#' @param features A vector of feature column names (default: X1 to X5).
#' @param target The target column name (default: "Country").
#' @param folds Number of cross-validation segments (default: 10).
#'
#' @return A list containing the trained PLS model and the original class levels.
trainPlsClassifier <- function(trainData, features = c("X1", "X2", "X3", "X4", "X5"), target = "Country", folds = 10) {
  
  # Dummy Matrix
  formula_y <- as.formula(paste("~", target, "- 1"))
  trainResponse <- model.matrix(formula_y, data = trainData)
  
  # Feature formula
  formula <- as.formula(paste("trainResponse ~", paste(features, collapse = " + ")))
  
  # Training model (PLS)
  plsModel <- plsr(
    formula, 
    data = trainData, 
    scale = TRUE, 
    validation = "CV", 
    segments = folds
  )
  
  return(list(
    model = plsModel,
    class_levels = levels(trainData[[target]])
  ))
}

#' Predict Classes using PLS Model
#'
#' @description 
#' A helper function to convert raw PLS numeric predictions back into discrete factor classes.
#'
#' @param plsObj The list returned by `trainPlsClassifier`.
#' @param newData The data.frame to predict on.
#' @param ncomp The number of components to use for prediction (default: 5).
#'
#' @return A factor vector of predicted classes.
predictPls <- function(plsObj, newData, ncomp = 5) {
  
  rawPred <- predict(plsObj$model, newdata = newData, ncomp = ncomp)
  
  classIdx <- apply(rawPred, 1, which.max)
  
  predictedClasses <- factor(plsObj$class_levels[classIdx], levels = plsObj$class_levels)
  
  return(predictedClasses)
}

#' Train Support Vector Machine (SVM) Classifier
#'
#' @description 
#' Automatically tunes the cost parameter and trains an SVM model using the optimal parameter.
#'
#' @param trainData The training dataset (data.frame).
#' @param features A vector of feature column names (default: X1 to X5).
#' @param target The target column name (default: "Country").
#' @param kernelType The type of SVM kernel: "linear", "radial", "polynomial", or "sigmoid".
#'
#' @return A trained SVM model object.
trainSvmClassifier <- function(trainData, features = c("X1", "X2", "X3", "X4", "X5"), target = "Country", kernelType = "polynomial") {
  
  formula <- as.formula(paste(target, "~", paste(features, collapse = " + ")))
  
  # Optimal cost parameter
  tuneResult <- tune(
    svm, formula, 
    data = trainData, 
    ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)),
    kernel = kernelType
  )
  bestCost <- tuneResult$best.parameters$cost
  
  # Training model (SVM)
  svmModel <- svm(
    formula, 
    data = trainData, 
    kernel = kernelType,  
    cost = bestCost,            
    scale = TRUE
  )
  
  return(svmModel)
}


#' Train Random Forest Classifier
#'
#' @description 
#' Trains a Random Forest model for multi-class classification.
#'
#' @param trainData The training dataset (data.frame).
#' @param features A vector of feature column names (default: X1 to X5).
#' @param target The target column name (default: "Country").
#'
#' @return A trained Random Forest model object.
trainRfClassifier <- function(trainData, features = c("X1", "X2", "X3", "X4", "X5"), target = "Country") {
  
  formula <- as.formula(paste(target, "~", paste(features, collapse = " + ")))
  
  rfModel <- randomForest(
    formula, 
    data = trainData,
    ntree = 500,        
    mtry = sqrt(length(features)), 
    importance = TRUE
  )
  
  return(rfModel)
}


#' Train Regularized Regression (Ridge / LASSO)
#'
#' @description 
#' Encapsulates the matrix conversion required by glmnet, performs 10-fold CV to find 
#' the optimal lambda, and returns the trained regularized regression model.
#'
#' @param trainData The training dataset (data.frame).
#' @param features A vector of feature column names.
#' @param target The target column name.
#' @param alphaVal The penalty type: 0 for Ridge, 1 for LASSO.
#'
#' @return A list containing the trained `cv.glmnet` model and the `optimal_lambda`.
trainRegularizedRegression <- function(trainData, features = c("X1", "X2", "X3", "X4", "X5"), target = "Country", alphaVal = 1) {
  
  train_X <- as.matrix(trainData[, features])
  train_Y <- as.factor(trainData[[target]])
  
  model <- cv.glmnet(
    train_X, train_Y, 
    alpha = alphaVal, 
    family = "multinomial", 
    nfolds = 10
  )
  
  return(list(
    model = model,
    optimalLambda = model$lambda.min
  ))
}