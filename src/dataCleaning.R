#' Generate Mixed Rice Samples
#'
#' @description 
#' Generates simulated mixed rice data by randomly sampling from the pure Taiwan 
#' and Vietnam rice datasets for a specified number of times. The features are 
#' combined using a randomized blending ratio (10% to 90%).
#'
#' @param data The raw dataset (Note: Currently, the function relies on global variables `vietnam` and `taiwan`; this parameter is inactive).
#' @param times The total number of mixed samples to generate (integer).
#'
#' @return A data.frame containing the blended feature values, along with the newly added `Country` ("Mixed") and ratio columns.
#' 
generateMixedSample <- function(data, times){
  
  mixed_data <- data.frame() # Create an empty dataframe to store mixed data
  
  for (i in seq_len(times)) {
    vietnam_sample <- vietnam[sample(seq_len(70), 1), ] # Randomly sample Vietnam rice from raw data
    taiwan_sample <- taiwan[sample(seq_len(80), 1), ]   # Randomly sample Taiwan rice from raw data
    
    ratio_t <- sample(seq_len(9), 1)/10 # Randomly generate the ratio of Taiwan rice (0.1 - 0.9)
    ratio_v <- 1 - ratio_t # Calculate the ratio of Vietnam rice
    
    vietnam_values <- as.numeric(vietnam_sample[, 2:6]) 
    taiwan_values <- as.numeric(taiwan_sample[, 2:6])
    
    # Calculate the blended feature values
    mixed_values <- round(vietnam_values * ratio_v + taiwan_values * ratio_t, 3) 
    
    mixed_sample <- data.frame(t(mixed_values))
    
    # Label the country as "Mixed" and bind the ratio columns
    mixed_sample <- cbind(mixed_sample, Country = "Mixed", ratio_t, ratio_v)
    
    # Combine the newly generated mixed sample into the main dataframe
    mixed_data <- rbind(mixed_data, mixed_sample)
  }
  
  return(mixed_data)  
}


#'
#' Prepare Training and Test Sets
#'
#' @description 
#' Randomly splits the cleaned data-set into training and test sets based on the 
#' specified ratio. It also automatically converts the target variable (`Country`) 
#' into a Factor type to meet the strict requirements of classification models 
#' (e.g., Random Forest, SVM).
#'
#' @param cleanData The complete data.frame containing both pure and mixed rice samples.
#' @param trainRatio The proportion of the data to be used for the training set (default is 0.8).
#'
#' @return A list containing two elements: `train` (the training data.frame) and `test` (the testing data.frame).
#' 
prepareTrainingSet <- function(cleanData, trainRatio = 0.8){
  
  # Ensure the target variable is a factor to prevent modeling errors
  if("Country" %in% colnames(cleanData)){
    cleanData$Country <- as.factor(cleanData$Country)
  }
  
  # Get the total number of rows and calculate the required training set size
  n_total <- nrow(cleanData)
  n_train <- round(n_total * trainRatio)
  
  # Randomly sample row indices for the training set (prevents data imbalance from sequential splitting)
  train_indices <- sample(seq_len(n_total), n_train)
  
  # Split the data based on the sampled indices
  trainData <- cleanData[train_indices, ]
  testData <- cleanData[-train_indices, ]
  
  return(list(
    train = trainData,
    test = testData
  ))
}