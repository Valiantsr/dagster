import os
import mlflow
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import requests

# Set up MLflow tracking
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs

# Load validation data
valid_url = "https://dagshub.com/api/v1/repos/valiant.shabri/dagster/storage/raw/s3/dagster/data/validate.csv"
valid_path = 'datasets/validate.csv'

if not os.path.exists(valid_path):
    response = requests.get(valid_url)
    with open(valid_path, 'wb') as f:
        f.write(response.content)

valid_data = pd.read_csv(valid_path)
valid_texts = valid_data['text'].tolist()

# Map string labels to integers
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
valid_labels = [label_mapping[label] for label in valid_data['label'].tolist()]

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=2)

# Load model from MLflow
model_uri = f"models:/IndoBERT-Sentiment-Model/latest"
model = mlflow.pytorch.load_model(model_uri)

# Create DataLoader
valid_dataset = SentimentDataset(valid_texts, valid_labels, tokenizer)
valid_loader = DataLoader(valid_dataset, batch_size=16)

# Evaluate model
model.eval()
total, correct = 0, 0

with mlflow.start_run(run_name="IndoBERT_Sentiment_Validation"):
    with torch.no_grad():
        for batch in valid_loader:
            outputs = model(**batch)
            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == batch['labels']).sum().item()
            total += batch['labels'].size(0)
        
        accuracy = correct / total
        mlflow.log_metric("validation_accuracy", accuracy)

print(f"Validation Accuracy: {accuracy:.4f}")
