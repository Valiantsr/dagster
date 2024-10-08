import os
import mlflow
import torch
import pandas as pd
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

# Set up MLflow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

# Load model from MLflow model registry
model_name = "SentimentAnalysisNLP"
client = mlflow.tracking.MlflowClient()

# Get the latest version of the model
latest_version = client.get_latest_versions(model_name, stages=["None"])[-1].version

# Prepare the directory where the model files will be saved
model_dir = '/tmp/sentiment_analysis_model'
os.makedirs(model_dir, exist_ok=True)

# Download and extract the model artifacts from MLflow
run_id = client.get_latest_versions(model_name)[-1].run_id
artifact_uri = client.download_artifacts(run_id, 'model', dst_path=model_dir)

# Log the directory for verification
print("Contents of model_dir:")
for root, dirs, files in os.walk(model_dir):
    for file in files:
        print(os.path.join(root, file))

# Load the tokenizer and model from the downloaded artifacts
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Download dataset to the same directory as model artifacts
url = "https://dagshub.com/api/v1/repos/valiant.shabri/dagster/storage/raw/s3/dagster/data/retrain.csv"
local_path = os.path.join(model_dir, 'retrain.csv')

if not os.path.exists(local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as f:
        f.write(response.content)
    print(f"Dataset downloaded and saved to: {local_path}")
else:
    print(f"Dataset already exists at: {local_path}")

# Load dataset
data = pd.read_csv(local_path)
texts = data['text'].tolist()
labels = data['label'].tolist()

# Prepare inputs using the tokenizer
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
labels = torch.tensor(labels)

# Fine-tune model
model.train()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
loss.backward()

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)
optimizer.step()

# Log the retrained model to MLflow
with mlflow.start_run(run_name="retrained_sentiment_model"):
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=model_name
    )
    mlflow.log_metric("training_loss", loss.item())
    mlflow.log_param("learning_rate", 1e-5)

print("Model retrained and logged successfully.")
