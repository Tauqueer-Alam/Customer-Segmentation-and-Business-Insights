import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def train():
    print("Loading data...")
    dataset = pd.read_csv('Mall_Customers.csv')
    
    # Preprocess
    dataset_cleaned = dataset.drop('CustomerID', axis=1)
    dataset_cleaned['Gender'] = dataset_cleaned['Gender'].map({
        'Male': 1,
        'Female': 0
    })
    
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(dataset_cleaned)
    
    # Save the fitted scaler
    print("Saving scaler artifact...")
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Training complete! scaler.pkl successfully saved.")

if __name__ == "__main__":
    train()
