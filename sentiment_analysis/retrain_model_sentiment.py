import os
import mlflow
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

# Load dataset
data = pd.read_csv('sentiment_analysis/datasets/train_data.csv')
texts = data['text'].tolist()
labels = data['label'].tolist()

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('model')
model = BertForSequenceClassification.from_pretrained('model')

# Prepare inputs
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
labels = torch.tensor(labels)

# Fine-tune model
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
        python_model=SentimentAnalysisModel(),
        registered_model_name="SentimentAnalysisNLP"
    )
    mlflow.log_metric("training_loss", loss.item())
    mlflow.log_param("learning_rate", 1e-5)

print("Model retrained and logged successfully.")
