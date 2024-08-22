import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Generate training data similar to the initial data
def generate_training_data():
    X_train, y_train = make_classification(
        n_samples=700, n_features=10, n_informative=2, n_redundant=8, 
        weights=[0.9, 0.1], flip_y=0, random_state=42
    )
    training_data = pd.DataFrame(X_train, columns=[f'Feature{i+1}' for i in range(X_train.shape[1])])
    training_data['Target'] = y_train
    training_data.to_csv('training_data.csv', index=False)

# Generate testing data similar to the initial data
def generate_testing_data():
    X_test, y_test = make_classification(
        n_samples=300, n_features=10, n_informative=2, n_redundant=8, 
        weights=[0.9, 0.1], flip_y=0, random_state=42
    )
    testing_data = pd.DataFrame(X_test, columns=[f'Feature{i+1}' for i in range(X_test.shape[1])])
    testing_data['Target'] = y_test
    testing_data.to_csv('testing_data.csv', index=False)

if __name__ == "__main__":
    generate_training_data()
    generate_testing_data()
