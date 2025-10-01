import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    roc_curve, ConfusionMatrixDisplay
)

def logistic_regression(df):
    """
    Logistic Regression training and evaluation with class imbalance handling.

    Steps:
    - Ask user for target variable.
    - If positives <25%, use 80/20 split with class_weight='balanced'.
    - Else, ask user for train/test ratio and fit normal logistic regression.
    - Print model summary, evaluation metrics, confusion matrix plot, and ROC curve.

    Args:
        df (pd.DataFrame): Preprocessed and transformed dataset.

    Returns:
        model (LogisticRegression): Trained logistic regression model.
    """

    # Step 1 - Ask for target variable
    print("Available columns:", df.columns.tolist())
    target_col = input("Enter the target variable: ").strip()
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Step 2 - Check class balance
    positive_ratio = y.mean()
    print(f"\nProportion of positives (1s): {positive_ratio:.2%}")

    if positive_ratio < 0.25:
        print("Detected imbalance (<25% positives). Using 80/20 split with class weights.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    else:
        ratio = input("Enter train split ratio (e.g., 0.7 for 70% train): ").strip()
        try:
            ratio = float(ratio)
            if not (0 < ratio < 1):
                raise ValueError
        except ValueError:
            print("Invalid ratio entered. Defaulting to 0.8 (80/20 split).")
            ratio = 0.8

        test_size = 1 - ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        model = LogisticRegression(max_iter=1000, solver="lbfgs")

    # Step 3 - Train model
    model.fit(X_train, y_train)
    print("\nModel training complete.")

    # Step 4 - Model summary
    coef_summary = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    print("\nModel Summary (Coefficients):")
    print(coef_summary)
    print(f"\nIntercept: {model.intercept_[0]:.4f}")

    # Step 5 - Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nEvaluation Metrics on Test Set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Step 6 - Plots
    # Confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()

    return model
