![UTA-DataScience-Logo](https://github.com/user-attachments/assets/6d626bcc-5430-4356-927b-97764939109d)


# Bank Customer Churn Prediction

* This repository holds an attempt to apply Machine Learning model for predicting bank customer churn using data from "Binary Classification with a Bank Churn" Kaggle Challenge https://www.kaggle.com/competitions/playground-series-s4e1/overview

## Overview

  * **Definition of the tasks / challenge**: The task, as defined by the Kaggle challenge, is to predict whether a bank customer will continue with their account or close it (churn). This binary classification problem is crucial for banks to proactively identify customers at risk of leaving and take preventive measures
  * **My approach**: The approach in this repository formulates the problem as classification task, using multiple machine learning models with bank customer features as input. Performance evaluation using multiple metrics
  * **Summary of the performance achieved**: Based on the ROC curves and classification reports, our models achieved the following performance on the validation set:
     * XGBoost: Best overall performance with ROC-AUC of 0.881 and Accuracy Score of 0.885
     * Random Forest: Close second with ROC-AUC of 0.869 and Accuracy Score of 0.880
     * Logistic Regression: Baseline model with reasonable performance with ROC-AUC of 0.797 and Accuracy Score of 0.859

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
    * Training: 78,194 customers
    * Validation: 26,065 customers
    * Testing: 26,065 customers


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

* **Credit Score Distribution**:
  * X-axis: Shows credit scores ranging from approximately 400 to 800
  * Y-axis: Shows density
  * The distribution shows:
    * Peak concentration around 650-700
    * Very few customers have scores below 500 or above 800
    * Both zero and non-zero balance customers have similar distributions
![Screenshot 2024-12-10 145403](https://github.com/user-attachments/assets/542daac6-de44-4248-8455-b534c084eaae)


* **Before and After Scaling Credit Score**:
 
  * X-axis: standard deviations (z-scores)
  * Y-axis: shows the raw count of customers
  * The distribution shows:
    * The new range is approximately from -4 to 2 standard deviations
    * Most values fall between -2 and +2 ( which is expected in normal distribution)
    * The new distribution is centered around 0
    ![Screenshot 2024-12-10 150643](https://github.com/user-attachments/assets/095dbde9-c02a-4b27-b6e9-95253562e494)
    ![Screenshot 2024-12-10 150719](https://github.com/user-attachments/assets/bad1d967-12ef-41b2-91bd-e3941cf097ca)


### Problem Formulation

  * **Input**:
    * Processed numerical features (8 columns)
    * Encoded categorical features (4 columns after one-hot encoding)
  * **Output**: 
    * Probability scores for churn prediction
   
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

* **Hardware and Software Environment**:
  * The code runs efficiently on standard CPU hardware
  * No GPU acceleration required
    
* **Training Duration**:
  * Approximately 1~2 minutes

* **Training Decisions**:
  * Models were trained without extensive hyperparameter tuning
  * No early stopping was used as training times were short
  

### Performance Comparison

* **Key Metrics**:
  * Accuracy Score
  * ROC-AUC Score
  * Classfication Report (Precision, Recall, F1-Score)
* **Result Visualization**:
  * ROC curves:
    ![Screenshot 2024-12-11 011831](https://github.com/user-attachments/assets/cc931be2-577c-466d-80c4-cc80f6a89a14)

  * Comparison table:
    ![Screenshot 2024-12-11 011600](https://github.com/user-attachments/assets/393a10ef-4f27-4031-8aca-208459f3f733)

 

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

* **Dataset Requirements**:
  * train.csv: Training dataset containing customer information and churn labels
  * test.csv: Testing dataset containing customer information but without churn labels

* **Results Reproduction**:
  * **1. Data Loading**
       * Load training and test datasets using pandas
       * Initial data exploration through visualizations and Initial Look

  * **2. Data Cleaning**
       * Handling missing values for both features
       * Addresing outliers
       * Remove uneccesary columns
         
  * **3. Feature Preprocessing**
       * Scale numerical features using StandardScaler
       * Encode categorical features (Geography, Gender) using OneHotEncoder
       * Combine scaled numerical and encoded categorical features

  * **4. Model Training and Evaluation**
       * Split data into train (60%), validation (20%), and test (20%) sets
       * Train three models:
         * Logistic Regression
         * Random Forest
         * XGBoost


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

### Dataset

* **Download train.csv and test.csv dataset on Kaggle**:
  * https://www.kaggle.com/competitions/playground-series-s4e1/data?select=train.csv
* **Loading original dataset (zip files)**:
  
  * **Upload the both datasets**:
    ```sh
    from google.colab import files
    uploaded = files.upload()
    ```
  * **Unzip datasets**:
    ```sh
    !unzip train.csv.zip -d /content/data/
    !unzip test.csv.zip -d /content/data/
    ```
 * **Loading original dataset (csv files)**:
   ```sh
   !wget https://drive.google.com/uc?id=1HTOP1TUVCkKaJRwO7N0vmnKDVjFmYXCr -O train.csv

   !wget https://drive.google.com/uc?id=1wfHyaS-gxuof8hd09hqAoGZVtl68lSH1 -O test.csv
   ```

   
## Citations

* https://www.kaggle.com/competitions/playground-series-s4e1/data?select=train.csv
* https://www.geeksforgeeks.org/advantages-and-disadvantages-of-logistic-regression/
* https://www.geeksforgeeks.org/what-are-the-advantages-and-disadvantages-of-random-forest/
* https://www.investopedia.com/terms/c/cost-benefitanalysis.asp#:~:text=A%20cost%2Dbenefit%20analysis%20(CBA,and%20subtracting%20the%20associated%20costs.
* https://stripe.com/resources/more/churn-analysis-101-a-how-to-guide-for-businesses#:~:text=Time%20series%20analysis%3A%20Examine%20how,the%20effect%20of%20new%20competitors.







