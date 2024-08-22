import pandas as pd
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# Set up MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/valiant.shabri/dagster.mlflow")

# Specify the name of the registered model you want to retrain
model_name = "your_existing_model_name"
model_version = 1  # Specify the version you want to load and retrain

# Load the generated training data
training_data = pd.read_csv('training_data.csv')
X_train = training_data.drop('Target', axis=1)
y_train = training_data['Target']

# Load the existing model from MLflow Model Registry
model_uri = f"models:/{model_name}/{model_version}"
loaded_model = mlflow.xgboost.load_model(model_uri)

# Retrain the model
loaded_model.fit(X_train, y_train)

# Start an MLflow run to log the retrained model
with mlflow.start_run(run_name="retrained_model") as run:
    # Log the retrained model
    mlflow.xgboost.log_model(loaded_model, "model")

    # Optionally, log additional parameters, metrics, or artifacts
    mlflow.log_param("retraining_version", model_version + 1)

    # Register the retrained model as a new version in the Model Registry
    client = MlflowClient()
    model_uri = f"runs:/{run.info.run_id}/model"
    client.create_model_version(name=model_name, source=model_uri, run_id=run.info.run_id)

# Output the registered model URI
print(f"Model retrained and logged as a new version in the Model Registry.")
