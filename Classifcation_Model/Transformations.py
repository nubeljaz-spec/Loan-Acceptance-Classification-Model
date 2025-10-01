import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def transform_data(df):
    """
    Detects numeric non-binary columns in a DataFrame and applies a user-selected transformation.
    Also allows the user to drop unwanted columns (e.g., ID, ZIP).
    
    Options:
        1 - Standardization (z-score)
        2 - Min-Max Scaling
        3 - Log Transformation
    
    Parameters:
        df (pd.DataFrame): Input DataFrame (after preprocessing)
    
    Returns:
        pd.DataFrame: Transformed DataFrame with optional column removal and scaling
    """

    df_transformed = df.copy()

    # Step 1 - Ask user for column removal
    print("Available columns:", df_transformed.columns.tolist())
    print("\nHint: Columns like 'id', 'ZIP' may not be useful for modeling.")
    cols_to_remove = input(
        "Enter column names to remove (comma-separated), or press Enter to keep all: "
    ).strip()

    if cols_to_remove:
        cols_to_remove = [col.strip() for col in cols_to_remove.split(",")]
        missing = [col for col in cols_to_remove if col not in df_transformed.columns]
        if missing:
            print(f"Warning: These columns not found in DataFrame: {missing}")
        cols_to_remove = [col for col in cols_to_remove if col in df_transformed.columns]
        df_transformed = df_transformed.drop(columns=cols_to_remove, errors="ignore")
        print(f"Removed columns: {cols_to_remove}")

    # Step 2 - Detect numeric columns excluding binary (0/1) ones
    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
    non_binary_cols = [
        col for col in numeric_cols
        if df_transformed[col].nunique() > 2
    ]

    if not non_binary_cols:
        print("No numeric non-binary columns detected for transformation.")
        return df_transformed

    print(f"\nDetected numeric columns for transformation: {non_binary_cols}")

    # Step 3 - Ask user for transformation method
    print("\nChoose a transformation method for numeric columns:")
    print("1 - Standardization (z-score)")
    print("2 - Min-Max Scaling")
    print("3 - Log Transformation")
    choice = input("Enter choice (1/2/3) or press Enter to skip: ").strip()

    if choice not in ["1", "2", "3"]:
        print("No transformation applied.")
        return df_transformed

    if choice == "1":
        scaler = StandardScaler()
        df_transformed[non_binary_cols] = scaler.fit_transform(df_transformed[non_binary_cols])
        print("Applied Standardization (z-score).")

    elif choice == "2":
        scaler = MinMaxScaler()
        df_transformed[non_binary_cols] = scaler.fit_transform(df_transformed[non_binary_cols])
        print("Applied Min-Max Scaling.")

    elif choice == "3":
        for col in non_binary_cols:
            df_transformed[col] = np.log1p(df_transformed[col])  # log(1+x) to avoid -inf at 0
        print("Applied Log Transformation.")

    return df_transformed
