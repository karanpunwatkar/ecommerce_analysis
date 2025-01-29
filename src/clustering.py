import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_loader import load_customers

def train_kmeans(df, features, n_clusters=3):
    """Trains a KMeans clustering model and adds cluster labels to the DataFrame."""
    # Check if the specified columns exist
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the dataset: {', '.join(missing_cols)}")
    
    # Convert categorical data to numeric if necessary (e.g., using label encoding for 'Region')
    if 'Region' in features:
        df['Region'] = df['Region'].astype('category').cat.codes
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
    
    model = KMeans(n_clusters=n_clusters)
    df['Cluster'] = model.fit_predict(df_scaled)
    
    return model, df

if __name__ == "__main__":
    # Load data
    df_customers = load_customers()
    
    # Check and clean column names
    df_customers.columns = df_customers.columns.str.strip()  # Clean any leading/trailing spaces
    print("Columns in df_customers:", df_customers.columns)  # For debugging purposes
    
    # Use available features, such as 'Region'
    features = ["Region"]  # Replace with available columns if needed
    
    # Train KMeans model
    try:
        model, clustered_df = train_kmeans(df_customers, features)
        print("Clustered Data:\n", clustered_df)
    except ValueError as e:
        print(f"Error: {e}")
