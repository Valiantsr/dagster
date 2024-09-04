import os
import mlflow
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
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

# Load training data
train_url = "https://dagshub.com/api/v1/repos/valiant.shabri/dagster/storage/raw/s3/dagster/data/train.csv"
train_path = 'datasets/train.csv'

if not os.path.exists(train_path):
    response = requests.get(train_url)
    with open(train_path, 'wb') as f:
        f.write(response.content)

train_data = pd.read_csv(train_path)
train_texts = train_data['text'].tolist()

# Map string labels to integers
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
train_labels = [label_mapping[label] for label in train_data['label'].tolist()]

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=len(label_mapping))

# Create DataLoader
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Fine-tune model
model.train()
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3
losses = []

# Ensure any active run is ended before starting a new one
# mlflow.end_run()

with mlflow.start_run(run_name="IndoBERT_Sentiment_Training"):
    for epoch in range(epochs):  # 3 epochs for demonstration
        epoch_loss = 0
        for batch in train_loader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(train_loader))
        mlflow.log_metric("train_loss", epoch_loss / len(train_loader), step=epoch)

    # Log hyperparameters to MLflow
    mlflow.log_params({
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": epochs,
        "model_name": "indobenchmark/indobert-base-p1",
    })
    # Log model to MLflow and register it in the model registry
    model_info = mlflow.pytorch.log_model(model, artifact_path="model")
    
    # Register the model
    mlflow.register_model(model_uri=model_info.model_uri, name="IndoBERT-Sentiment-Model")

print("Training complete and model registered.")
