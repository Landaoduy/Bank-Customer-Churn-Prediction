![](UTA-DataScience-Logo.png)

# Bank Customer Churn Prediction

* This repository holds an attempt to apply Machine Learning model for predicting bank customer churn using data from "Binary Classification with a Bank Churn" Kaggle Challenge https://www.kaggle.com/competitions/playground-series-s4e1/overview

## Overview

  * **Definition of the tasks / challenge**: The task, as defined by the Kaggle challenge, is to predict whether a bank customer will continue with their account or close it (churn). This binary classification problem is crucial for banks to proactively identify customers at risk of leaving and take preventive measures
  * **My approach**: 
  * **Summary of the performance achieved**: Based on the ROC curves and classification reports, our models achieved the following performance on the validation set:
     * XGBoost: Best overall performance with ROC-AUC of 0.871884 and Accuracy Score of 0.879514
     * Random Forest: Close second with ROC-AUC of 0.859379 and Accuracy Score of 0.875680
     * Logistic Regression: Baseline model with reasonable performance with ROC-AUC of 0.784249 and Accuracy Score of 0.859043

## Summary of Workdone

### Data

* Data:
  * **Type**: Tabular Dataset
    * Input: Train and Test CSV file of the bank's customer features
    * Output: The probability for the target variable 'Exited'
  * **Size**:
    * 165,034 rows and 14 columns for Train dataset
    * 110,023 rows and 13 columns for Test dataset
  * **Instances (Train, Test, Validation Split)**:
    * Training: 78,256 customers
    * Validation: 26,086 customers
    * Testing: 26,086 customers


#### Data Cleaning

* Handle missing values for numerical columns
  * **Numerical Columns**:
    * Replace NA values with median values
  * **Categorical Columns**:
    * Filled missing values with mode (most frequent value)
      
* Outlier Treatment
   * **Age**: Clipped values to range [18, 100], replaced outliers with median
   * **Balance**: Applied upper and lower bounds based on statistical distribution
   * **Tenure**: Replaced negative values with median

#### Preprocessing

* **Separate numerical and categorical features**:
  * Categorical features: Geography, Gender
  * Numerical features: Age, Balance, Tenure, ...
    
* **Numerical features processing**:
  * Applied StandardScaler(Scikit learn) for normalization
  * Ensured all features are on the same scale
    
* **Categorical features processing**:
  * Use OneHotEncoder (Scikit learn)
  * Create binary columns for each category


#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

  * **Input**:
    * Processed numerical features (8 columns)
    * Encoded categorical features (4 columns after one-hot encoding)
  * **Output**:
   
  * **Models**
    * **Logistic Regression**:
      * **Parameters**:
        * max_iter = 500 : Maximum number of iterations to converge
      * **Advantages**:
        * Simple and very efficient to train
        * Good for understanding baseline performance
      * **Disadvantages**:
        * Cannot capture non-linear relationships between features
        * Assumes independence between features
          
    * **Random Forest**:
      * **Parameters**:
        * n_estimators = 100: Number of trees
        * random_state = 42: Ensure reproducibility
      * **Advantages**:
        * Handles both numerical and categorical features
        * Can handle missing values and outliers effectively
      * **Disadvantages**:
        * Longer prediction time compare to other models => May require more memory
        * Can overfit if trees are too deep
    * **XGBoost**:
      * **Parameters**:
        * eval_metric = 'logloss': Uses log loss for evaluation
        * randome_state = 42
      * **Advantages**:
        * Usually achieve high accuracy and performance
        * Handles imbalance data well
      * **Disadvantages**:
        * Can be overfitting
        * Can be memory-intensive, especially for large datasets.
       
      
### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* **Key Metrics**:
  * Accuracy Score
  * ROC-AUC Score
  * Classfication Report (Precision, Recall, F1-Score)
* **Result Visualization**:
  * ROC curves:
  ![Screenshot 2024-12-09 182531](https://github.com/user-attachments/assets/e4ab70de-6a54-459d-9147-d157cd4daeef)

  * Comparison table:
  ![Screenshot 2024-12-09 181736](https://github.com/user-attachments/assets/532352ac-0de7-4f21-8c2a-6265ed15bc4e)


### Conclusions

* **Model Performance**:
  * XGBoost shows best overall performance
  * Random Forest come to second with similar metrics
  * Logistic Regression provides good baseline
    
### Future Work

* **Model Improvements**:
  * Implement neural networks
  * Try more advanced ensemble methods
* **Additional Analysis**:
  * Time-series analysis of churn patterns
  * Cost-benefit analysis of predictions

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
* **Required packages**:
  * Pandas, NumPy, Scikit-learn, XGBoost, matplotplib, seaborn
* **Installing packages in Jupyter/Colab**:
  * ```sh
    !pip install pandas
    !pip install numpy
    !pip install scikit-learn
    !pip install xgboost
    !pip install matplotlib
    !pip install seaborn
    ``` 

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







