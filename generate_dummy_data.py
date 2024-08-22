import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate dummy training and testing data based on your original data generation method
def generate_data():
    np.random.seed(42)
    # Create an imbalanced binary classification dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, 
                               weights=[0.9, 0.1], flip_y=0, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Create training DataFrame
    training_data = pd.DataFrame(X_train, columns=[f'Feature{i+1}' for i in range(X_train.shape[1])])
    training_data['Target'] = y_train
    
    # Create testing DataFrame
    testing_data = pd.DataFrame(X_test, columns=[f'Feature{i+1}' for i in range(X_test.shape[1])])
    testing_data['Target'] = y_test
    
    # Save to CSV
    training_data.to_csv('training_data.csv', index=False)
    testing_data.to_csv('testing_data.csv', index=False)

if __name__ == "__main__":
    generate_data()
