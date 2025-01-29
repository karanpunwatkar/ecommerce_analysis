import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from data_loader import load_customers

def train_lookalike_model(df, features):
    """Finds similar customers using KNN based on selected features."""
    # Check if the specified columns exist
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the dataset: {', '.join(missing_cols)}")
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])

    model = NearestNeighbors(n_neighbors=5, algorithm="auto")
    model.fit(df_scaled)
    
    return model, scaler

def find_similar_customers(df, model, scaler, target_customer):
    """Finds similar customers to the given target customer."""
    target_scaled = scaler.transform([target_customer])
    distances, indices = model.kneighbors(target_scaled)
    
    return df.iloc[indices[0]]

if __name__ == "__main__":
    # Load data
    df_customers = load_customers()
    
    # Check and clean column names
    df_customers.columns = df_customers.columns.str.strip()  # Clean any leading/trailing spaces
    print("Columns in df_customers:", df_customers.columns)  # For debugging purposes
    
    # Use available features, such as 'Region'
    features = ["Region"]  # Or choose other relevant columns
    
    # Convert categorical data to numeric if necessary (e.g., using label encoding for 'Region')
    df_customers['Region'] = df_customers['Region'].astype('category').cat.codes
    
    # Train lookalike model
    try:
        model, scaler = train_lookalike_model(df_customers, features)
        similar_customers = find_similar_customers(df_customers, model, scaler, [1])  # Example: target customer from Region 1
        print("Similar Customers:\n", similar_customers)
    except ValueError as e:
        print(f"Error: {e}")
