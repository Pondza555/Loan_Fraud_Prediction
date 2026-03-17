# Loan Default Prediction using XGBoost

## 🔹 Introduction

This project builds a binary classifier to predict whether a home loan borrower will default, using a dataset of deterministic financial and demographic factors. The project performs EDA, data cleaning, encoding, and compares an XGBoost model (with hyperparameter tuning) against a baseline logistic regression model, evaluated by AUC-ROC score and Confusion Matrix.

**Main notebook:** `loan.ipynb`

## 🔹 Objectives

- Predict loan default (binary: default vs. non-default) for home loan applicants.
- Perform EDA and data cleaning on a multi-feature mortgage dataset.
- Maximize AUC-ROC score using XGBoost with hyperparameter tuning.
- Evaluate the model with Precision, Recall, and Confusion Matrix.
     
## 🔹 Dataset & Features
     
- **Source:** Kaggle — [Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset/data)
- **Records:** 148,670 rows × 34 columns
- **Target:** `Status` — loan default (1 = Default, 0 = Non-default)
- **Key Features:** loan_limit, Gender, approv_in_adv, loan_type, loan_purpose, Credit_Worthiness, open_credit, business_or_commercial, income, credit_type, Credit_Score, LTV, Region, Security_Type, dtir1
- **Class Distribution:** Imbalanced — defaults are significantly fewer than non-defaults
               
## 🔹 EDA Highlights
               
- No missing values across all 148,670 records after data loading.
- Distribution analysis on all categorical features (loan_purpose, credit_type, business_or_commercial, construction_type, occupancy_type, Security_Type, etc.).
- Label Encoding used for categorical variables (preferred over one-hot encoding to avoid model instability with many categories).
                     
## 🔹 Methodology
                     
- **Data Preparation**
  - Categorical features encoded with LabelEncoder.
  - Train/Test split: 80% / 20% (stratified).
                         
- **Modeling**
  - Baseline: Logistic Regression
  - Primary model: XGBoost with Bayesian Optimization (hyperparameter tuning via cross-validation)
  - Tuned hyperparameters: colsample_bytree, learning_rate, max_depth, n_estimators, subsample
                               
- **Evaluation**
  - AUC-ROC (primary metric), Precision, Recall, F1-score
  - Precision-Recall vs. Threshold curve for threshold selection
  - Confusion Matrix
                                     
## 🔹 Results
                                     
  - **Best Model: XGBoost (No SMOTE)**
                                     
  - | Class        | Precision | Recall | F1-score | Support |
  - |--------------|-----------|--------|----------|---------|
  - | 0 (Non-default) | 0.9997 | 0.9999 | 0.9998  | 33,609  |
  - | 1 (Default)     | 0.9998 | 0.9990 | 0.9994  | 10,992  |
  - | **Accuracy** | | | **0.9997** | 44,601 |
                                     
  - **Confusion Matrix:**
    - TN = 33,607 | FP = 2
    - FN = 11    | TP = 10,981
                                         
    - ✅ **AUC = 1.00 (near-perfect on test set)** — XGBoost achieved near-perfect classification.
    - ⚠️ Note: The near-perfect result suggests the dataset may be synthetically generated or labels are highly predictable; real-world performance should be validated with production data.
                                         
## 🔹 Conclusion
                                         
1. XGBoost produced near-perfect AUC and Confusion Matrix results on the test set.
2. The model missed only 13 predictions (2 FP + 11 FN).
3. The near-perfect result strongly suggests the data is synthetically generated.
4. Should use more testing data or production data to assess real-world generalizability.
5. For actual deployment, threshold adjustment is recommended to balance FP and FN based on business cost objectives.
                                                       
## 🔹 Executive Summary
                                                       
  This project predicts home loan defaults using a 148,670-record Kaggle dataset. After label encoding of categorical features, XGBoost with Bayesian hyperparameter tuning achieved near-perfect AUC (~1.00) with only 13 misclassifications on the test set. The extremely high performance is likely attributable to the synthetic nature of the dataset. Despite this, the end-to-end pipeline (EDA → encoding → XGBoost tuning → threshold analysis → Confusion Matrix) demonstrates a production-ready framework that can be applied to real loan default prediction with appropriate data and threshold calibration.
