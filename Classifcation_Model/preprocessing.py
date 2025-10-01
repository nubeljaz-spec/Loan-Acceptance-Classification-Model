import pandas as pd

def preprocess_data(df, max_unique=10):
    """
    Preprocess dataset: detect categorical variables, prompt user for labels,
    and one-hot encode them.

    Args:
        df (pd.DataFrame): Input dataset.
        max_unique (int): Max unique values to consider a numeric column categorical.

    Returns:
        pd.DataFrame: Processed dataframe with one-hot encoding.
    """

    df_processed = df.copy()
    categorical_columns = []

    # Detect categorical candidates
    for col in df.columns:
        unique_vals = df[col].unique()

        # Detect numeric categoricals
        if df[col].dtype in ['int64', 'float64'] and len(unique_vals) <= max_unique:

            # Skip binary variables
            if len(unique_vals) == 2:
                continue

            try:
                sorted_vals = sorted(int(v) for v in unique_vals)
            except ValueError:
                sorted_vals = sorted(unique_vals, key=str)

            print(f"\nDetected possible categorical variable: '{col}' with unique values {sorted_vals}")
            user_input = input(
                f"Enter labels for '{col}' categories in order ({len(sorted_vals)} labels, comma-separated), or leave blank to skip: "
            )

            if user_input.strip():
                labels = [label.strip() for label in user_input.split(",")]

                if len(labels) != len(sorted_vals):
                    raise ValueError(f"Number of labels provided does not match categories for '{col}'")

                mapping = dict(zip(sorted_vals, labels))
                df_processed[col] = df_processed[col].map(mapping)
                categorical_columns.append(col)
            else:
                print(f"Skipping '{col}' as categorical.")

        # Detect object/string categoricals
        elif df[col].dtype == 'object':
            categorical_columns.append(col)

    # One-hot encode detected categorical columns
    df_processed = pd.get_dummies(df_processed, columns=categorical_columns, drop_first=False, dtype=int)

    return df_processed
