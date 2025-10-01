# Loan Prediction Project

This project is a **machine learning toolkit** designed for predicting loan approvals (or other binary classification tasks) using **Logistic Regression** and **Random Forest** models.

# Loan Acceptance Prediction Pipeline

This repository provides a full pipeline for building and evaluating machine learning models for **loan acceptance prediction**.  

## Features

- **Preprocessing**  
  Detect categorical variables, encode them, and prepare the dataset.

- **Transformations**  
  Remove irrelevant columns, apply scaling or log transformations.

- **Logistic Regression**  
  Train and evaluate a logistic regression model with imbalance handling.

- **Random Forest**  
  Train and evaluate a random forest model with visualization.


It includes functions that:  
- Handle **class imbalance automatically** (important when your target variable has fewer than 25% positives).  
- Ask the user for **train/test split ratios** when data is balanced.  
- Provide **evaluation metrics and plots** (confusion matrix, ROC curve).  
- Allow **feature transformations** (standardization, scaling, log transforms).  
- Let users **remove irrelevant columns** like `ID`, `ZIP Code`, etc.  
- Plot a **Random Forest decision tree** (limited to 5 levels for readability).  

---

## Installation

Clone the repository and install the dependencies:

`bash
git clone https://github.com/nubeljaz-spec/Loan-Acceptance-Classification-Model.git
cd Loan-Acceptance-Classification-Model
pip install -r requirements.txt`

## Usage: You can import the functions from the provided script into your project. For example:

`import pandas as pd
from functions import preprocess_data, transform_data, logistic_regression, random_forest`
Load your dataset
`df = pd.read_csv("your_data.csv")`
Step 1: Preprocess
`df_preprocessed = preprocess_data(df)`
Step 2: Transform
`df_transformed = transform_data(df_preprocessed)`
Step 3: Logistic Regression
`logit_model = logistic_regression(df_transformed)`
Step 4: Random Forest
`rf_model = random_forest(df_transformed)`

### Function Details

#### 1. `preprocess_data(df, max_unique=10)`

**Input:**
- `df` → A pandas DataFrame.  
- `max_unique` → Maximum number of unique numeric values to consider as categorical (default: `10`).  

**Process:**
1. Detects categorical variables:
   - Numeric columns with limited unique values (`<= max_unique`).  
   - String/object-type columns.  
2. (Optional) Prompts the user to provide category labels.  
3. One-hot encodes categorical variables.  

**Output:**
- Returns a processed DataFrame ready for modeling.  

### 2. `transform_data(df)`

**Input:**
- `df` → Preprocessed DataFrame.  

**Process:**
1. Asks user which columns to remove (e.g., IDs, ZIP codes).  
2. Detects numeric non-binary columns.  
3. Prompts user to select a transformation:  
   - `1`: Standardization (z-score)  
   - `2`: Min-Max scaling  
   - `3`: Log transformation  

**Output:**
- Returns a transformed DataFrame ready for modeling.  
### 3. `logistic_regression(df)`

**Input:**
- `df` → Preprocessed and transformed DataFrame.  

**Process:**
1. Prompts user for target variable.  
2. Handles class imbalance (if positives < 25%).  
3. Splits train/test data (customizable).  
4. Fits a logistic regression model.  

**Displays:**
- Coefficients summary  
- Accuracy, Precision, Recall, F1, ROC-AUC  
- Confusion matrix (text + plot)  
- ROC curve  

**Output:**
- Returns a trained `LogisticRegression` model object.

  ### 4. `random_forest(df)`

**Input:**
- `df` → Preprocessed and transformed DataFrame.  

**Process:**
1. Prompts user for target variable.  
2. Handles class imbalance (if positives < 25%).  
3. Splits train/test data (customizable).  
4. Fits a Random Forest model.  

**Displays:**
- Feature importances  
- Accuracy, Precision, Recall, F1, ROC-AUC  
- Confusion matrix (text + plot)  
- ROC curve  
- Visualization of a single decision tree (first 5 levels)  

**Output:**
- Returns a trained `RandomForestClassifier` model object.  

