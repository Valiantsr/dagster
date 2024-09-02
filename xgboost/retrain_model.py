import pandas as pd
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report
import os

# Set environment variables for MLflow
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'

# Specify the name of the registered model you want to retrain
model_name = "XGB-Smote"
model_version = 2  # Specify the version you want to load and retrain

# Load the generated training data
training_data = pd.read_csv('training_data.csv')
X_train = training_data.drop('Target', axis=1)
y_train = training_data['Target']

# Load the existing model from MLflow Model Registry
model_uri = f"models:/{model_name}/{model_version}"
loaded_model = mlflow.xgboost.load_model(model_uri)

# Set additional XGBoost parameters and retrain the mode
# params = {"use_label_encoder": False, "eval_metric": 'logloss'}
loaded_model.set_params(eval_metric='logloss', use_label_encoder=False)
loaded_model.fit(X_train, y_train)

# Generate predictions and classification report
y_pred = loaded_model.predict(X_train)
report = classification_report(y_train, y_pred, output_dict=True)

# Start an MLflow run to log the retrained model and metrics
with mlflow.start_run(run_name="retrained_model") as run:
    # Log the retrained model
    mlflow.xgboost.log_model(loaded_model, "model")

    # Log retraining parameters
    # mlflow.log_param(params)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("model_version", model_version)
    mlflow.log_param("eval_metric", "logloss")
    mlflow.log_param("use_label_encoder", False)
    mlflow.log_param("retraining_version", model_version + 1)

    # Log classification report metrics
    mlflow.log_metrics({
        'accuracy': report['accuracy'],
        'recall_class_1': report['1']['recall'],
        'recall_class_0': report['0']['recall'],
        'f1_score_macro': report['macro avg']['f1-score']
    })

    # Register the retrained model as a new version in the Model Registry
    client = MlflowClient()
    model_uri = f"runs:/{run.info.run_id}/model"
    client.create_model_version(name=model_name, source=model_uri, run_id=run.info.run_id).version

# Output the registered model URI
print("Model retrained and logged as a new version in the Model Registry.")
