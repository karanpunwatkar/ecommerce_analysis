import pandas as pd

def clean_data(df):
    """Removes duplicates and fills missing values."""
    df = df.drop_duplicates()
    df = df.fillna(method="ffill")
    return df

def describe_data(df):
    """Returns basic statistics of the dataset."""
    return df.describe()

def check_missing_values(df):
    """Returns missing values count in each column."""
    return df.isnull().sum()

if __name__ == "__main__":
    # Load data and test EDA functions
    from data_loader import load_customers
    
    df_customers = load_customers()
    df_cleaned = clean_data(df_customers)

    print("Missing values before cleaning:\n", check_missing_values(df_customers))
    print("Missing values after cleaning:\n", check_missing_values(df_cleaned))
