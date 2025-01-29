import pandas as pd

def load_customers(file_path="data/Customers.csv"):
    """Loads customer data from a CSV file."""
    return pd.read_csv(file_path)

def load_products(file_path="data/Products.csv"):
    """Loads product data from a CSV file."""
    return pd.read_csv(file_path)

def load_transactions(file_path="data/Transactions.csv"):
    """Loads transaction data from a CSV file."""
    return pd.read_csv(file_path)

if __name__ == "__main__":
    # Test loading functions
    df_customers = load_customers()
    df_products = load_products()
    df_transactions = load_transactions()
    
    print("Data loaded successfully!")
