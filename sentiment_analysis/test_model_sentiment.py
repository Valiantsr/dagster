# import os
# import mlflow
# from transformers import BertTokenizer, BertForSequenceClassification
# import pandas as pd
# import requests
# import csv

# os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
# os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

# # URL untuk mengunduh test_data.csv dari DagsHub
# url = "https://dagshub.com/api/v1/repos/valiant.shabri/dagster/storage/raw/s3/dagster/data/test.csv"
# local_path = 'sentiment_analysis/datasets/test.csv'

# # Jika file belum ada di direktori lokal, unduh dari DagsHub
# if not os.path.exists(local_path):
#     response = requests.get(url)
#     os.makedirs(os.path.dirname(local_path), exist_ok=True)
#     with open(local_path, 'wb') as f:
#         f.write(response.content)

# # Load test dataset
# # col_names = ['text','label']
# test_data = pd.read_csv(local_path)
# texts = test_data['text'].tolist()
# true_labels = test_data['label'].tolist()

# # Load model
# # registered_model = mlflow.get_latest_registered_model("SentimentAnalysisNLP")
# model_uri = f"models:/SentimentAnalysisNLP/latest"
# model = mlflow.pyfunc.load_model(model_uri)

# tokenizer_local_path = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri}/artifacts/model_dir")
# # tokenizer_artifact_uri = f"{model_uri}/artifacts/tokenizer"
# tokenizer = BertTokenizer.from_pretrained(tokenizer_local_path)

# # Prepare inputs
# # tokenizer = BertTokenizer.from_pretrained('model')
# # tokenizer = model._model_impl.tokenizers
# inputs = tokenizer(texts, return_tensors="pt", padding=True)
# preds = model.predict(pd.DataFrame({'text': texts}))

# # Calculate accuracy
# accuracy = (preds == true_labels).mean()
# print(f"Test Accuracy: {accuracy}")

import os
import mlflow
from transformers import BertTokenizer
import pandas as pd
import requests
import torch
import time
from requests.exceptions import RequestException

# Setup MLflow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

# URL untuk mengunduh test_data.csv dari DagsHub
url = "https://dagshub.com/api/v1/repos/valiant.shabri/dagster/storage/raw/s3/dagster/data/test.csv"
local_path = 'sentiment_analysis/datasets/test.csv'

# Function to download file with retries
def download_file_with_retries(url, local_path, retries=5, timeout=10):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return
        except (RequestException, IOError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

# Jika file belum ada di direktori lokal, unduh dari DagsHub
if not os.path.exists(local_path):
    download_file_with_retries(url, local_path)

# Load test dataset
test_data = pd.read_csv(local_path)
texts = test_data['text'].tolist()
true_labels = test_data['label'].tolist()

# Load the logged model
model_uri = "models:/SentimentAnalysisNLP/latest"
model = mlflow.pyfunc.load_model(model_uri)

# Retry mechanism for downloading artifacts
def download_artifacts_with_retries(model_uri, retries=5, timeout=10):
    for attempt in range(retries):
        try:
            tokenizer_local_path = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri}/artifacts/model_dir")
            return tokenizer_local_path
        except mlflow.exceptions.MlflowException as e:
            print(f"Attempt {attempt + 1} to download artifacts failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

# Download tokenizer artifacts to a local directory with retries
tokenizer_local_path = download_artifacts_with_retries(model_uri)

# Load the tokenizer from the downloaded artifacts
tokenizer = BertTokenizer.from_pretrained(tokenizer_local_path)

# Prepare inputs using the loaded tokenizer
inputs = tokenizer(texts, return_tensors="pt", padding=True)
input_df = pd.DataFrame({"text": texts})

# Predict using the loaded model
preds = model.predict(input_df)

# Calculate accuracy
accuracy = (preds == true_labels).mean()
print(f"Test Accuracy: {accuracy}")
