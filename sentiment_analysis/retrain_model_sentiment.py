import os
import mlflow
import mlflow.pyfunc
import torch
import pandas as pd
import requests
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AlbertForSequenceClassification

# Set up MLflow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

# Load model from MLflow model registry
registered_model_name = "SentimentAnalysisNLP"
model_uri = f"models:/SentimentAnalysisNLP/latest"
loaded_model = mlflow.pyfunc.load_model(model_uri)
model = loaded_model._model_impl(loaded_model)
tokenizer = BertTokenizer.from_pretrained(loaded_model)

# if not os.path.exists(model_dir):
#     os.makedirs(model_dir, exist_ok=True)
#     model_artifact_uri = f"{model_uri}/artifacts/model"
#     mlflow.artifacts.download_artifacts(model_artifact_uri, dst_path=model_dir)

# model_dir = loaded_model._model_impl.get_model_meta().local_path
# model_dir = os.path.join(model_dir, "artifacts", "models")

# model_dir = model_uri

# if os.path.exists(model_dir):
#     print(f"Loading model from: {model_dir}")
#     print(f"Files in model directory: {os.listdir(model_dir)}")
    
#     # Load the model and tokenizer from the model directory
#     config = AutoConfig.from_pretrained(model_dir)
#     tokenizer = BertTokenizer.from_pretrained(model_dir)
#     model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
#     print("Model and tokenizer loaded successfully")
# else:
#     raise FileNotFoundError(f"Model directory not found: {model_dir}")

url = "https://dagshub.com/api/v1/repos/valiant.shabri/dagster/storage/raw/s3/dagster/data/retrain.csv"
local_path = 'sentiment_analysis/datasets/retrain.csv'

# Jika file belum ada di direktori lokal, unduh dari DagsHub
if not os.path.exists(local_path):
    response = requests.get(url)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, 'wb') as f:
        f.write(response.content)

# Load dataset
data = pd.read_csv(local_path)
texts = data['text'].tolist()
labels = data['label'].tolist()

# Prepare inputs using the tokenizer
# inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
# tokenizer = BertTokenizer.from_pretrained(model_dir)
# print(f"Loaded Tokenizer: {tokenizer}")
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
labels = torch.tensor(labels)

# Fine-tune model
# model = AutoModelForSequenceClassification.from_pretrained(model_dir)
# print(f"Loaded Model: {model}") 
model.train()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
loss.backward()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
optimizer.step()

# Log the retrained model
with mlflow.start_run(run_name="retrained_sentiment_model"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=loaded_model._model_impl,  # Use the model from the registry
        registered_model_name=registered_model_name,
        artifacts={"model_dir": model_dir}  # Correct path for saving artifacts
    )
    mlflow.log_metric("training_loss", loss.item())
    mlflow.log_param("learning_rate", 1e-5)

print("Model retrained and logged successfully.")