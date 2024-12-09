![](UTA-DataScience-Logo.png)

# Bank Customer Churn Prediction

* This repository holds an attempt to apply Machine Learning model for predicting bank customer churn using data from "Binary Classification with a Bank Churn" Kaggle Challenge https://www.kaggle.com/competitions/playground-series-s4e1/overview

## Overview

  * **Definition of the tasks / challenge**: The task, as defined by the Kaggle challenge, is to predict whether a bank customer will continue with their account or close it (churn). This binary classification problem is crucial for banks to proactively identify customers at risk of leaving and take preventive measures
  * **My approach**: 
  * **Summary of the performance achieved**: Based on the ROC curves and classification reports, our models achieved the following performance on the validation set:
     * XGBoost: Best overall performance with ROC-AUC of 0.871884 and Accuracy Score of 0.879514
     * Random Forest: Close second with ROC-AUC of 0.875680 and Accuracy Score of 0.859379
     * Logistic Regression: Baseline model with reasonable performance with ROC-AUC of 0.859043 and Accuracy Score of 0.784249

## Summary of Workdone

### Data

* Data:
  * Type: Tabular Dataset
    * Input: Train and Test CSV file of the bank's customer features
    * Output: The probability for the target variable 'Exited'
  * Size:
    * 165,034 rows and 14 columns for Train dataset
    * 110,023 rows and 13 columns for Test dataset
  * Instances (Train, Test, Validation Split):
    * Training: 78,256 customers
    * Validation: 26,086 customers
    * Testing: 26,086 customers


#### Data Cleaning

* Handle missing values for numerical columns
  1. Numerical Columns:
    * Replace NA values with median values
  2. Categorical Columns:
    * Filled missing values with mode (most frequent value)
      
* Outlier Treatment
   * Age: Clipped values to range [18, 100], replaced outliers with median
   * Balance: Applied upper and lower bounds based on statistical distribution
   * Tenure: Replaced negative values with median

#### Preprocessing

* Separate numerical and categorical features:
  * Categorical features: Geography, Gender
  * Numerical features: Age, Balance, Tenure, ...
* Scale features:
  * Import both StandardScaler and MinMaxScaler from Sci-kit learn library. However, I choose StandardScaler as my scaling strategy
* Encode categorical features:
  * Applied OneHotEncoder to convert categorical features into a binary format 
  * Value of 1 indicates the presence of that category, and 0 indicates its absence



#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







