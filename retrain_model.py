import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# Load training data
training_data = pd.read_csv('training_data.csv')
X_train = training_data.drop('Target', axis=1)
y_train = training_data['Target']

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Log model with MLflow
mlflow.sklearn.log_model(model, "model")
