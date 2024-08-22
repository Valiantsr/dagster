import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# Load testing data
testing_data = pd.read_csv('testing_data.csv')
X_test = testing_data.drop('Target', axis=1)
y_test = testing_data['Target']

# Load the model
model_name = "XGB-smote"
# production_model_name = "anomaly-detection-prod"
client = MlflowClient()
latest_version = client.get_latest_versions(model_name, stages=["None", "Production"])[-1]
model_uri = f"models:/{model_name}/{latest_version}"
model = mlflow.xgboost.load_model(model_uri)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy}")
