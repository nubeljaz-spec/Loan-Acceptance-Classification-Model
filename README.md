# Loan Prediction Project

This project is a **machine learning toolkit** designed for predicting loan approvals (or other binary classification tasks) using **Logistic Regression** and **Random Forest** models.

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

```bash
git clone https://github.com/your-username/loan-prediction-project.git
cd loan-prediction-project
pip install -r requirements.txt
