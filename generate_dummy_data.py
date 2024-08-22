import numpy as np
import pandas as pd

# Generate dummy training data
def generate_training_data():
    np.random.seed(42)
    X_train = np.random.rand(100, 4)
    y_train = np.random.randint(0, 2, 100)
    training_data = pd.DataFrame(X_train, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])
    training_data['Target'] = y_train
    training_data.to_csv('training_data.csv', index=False)

# Generate dummy test data
def generate_testing_data():
    np.random.seed(42)
    X_test = np.random.rand(20, 4)
    y_test = np.random.randint(0, 2, 20)
    testing_data = pd.DataFrame(X_test, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])
    testing_data['Target'] = y_test
    testing_data.to_csv('testing_data.csv', index=False)

if __name__ == "__main__":
    generate_training_data()
    generate_testing_data()
