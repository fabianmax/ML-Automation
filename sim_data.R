library(Xy)
library(caret)
library(dplyr)
library(data.table)

n_data_set <- 10

for (i in seq(n_data_set)) {
  
  # Sim settings
  n <- floor(runif(1, 500, 5000))
  n_num_vars <- c(sample(2:10, 1), sample(2:10, 1))
  n_cat_vars <- c(0, 0)
  n_noise_vars <- sample(1:5, 1)
  inter_degree <- sample(2:3, 1)
  
  # Simulate data
  sim <- Xy(n = n,   
            numvars = n_num_vars,
            catvars = n_cat_vars, 
            noisevars = n_noise_vars,   
            nlfun = function(x) {x^inter_degree},  
            interaction = 2,  
            sig = c(1,4),   
            cor = c(0.1, 0.3),  
            weights = c(-5,5),  
            noise.coll = FALSE)
  
  # Get data
  df <- sim$data
  dgp <- sim$dgp
  
  # Remove Intercept
  df[, "(Intercept)"] <- NULL
  
  # Rename columns
  names(df) <- gsub("(?<![0-9])0+", "", names(df), perl = TRUE)
  
  # Create test/train split
  df <- dplyr::rename(df, label = y)
  in_train <- createDataPartition(y = df$label, p = 0.7, list = FALSE)
  df_train <- df[in_train, ]
  df_test <- df[-in_train, ]
  
  # Export
  path_train <- paste0("../data/Xy/", i, "_train.csv")
  path_test <- paste0("../data/Xy/", i, "_test.csv")
  
  fwrite(df_train, file = path_train)
  fwrite(df_test, file = path_test)
  
}



