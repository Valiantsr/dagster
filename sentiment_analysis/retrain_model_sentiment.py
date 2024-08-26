# import os
# import mlflow
# from transformers import BertTokenizer, BertForSequenceClassification
# import pandas as pd
# import torch

# # Setup MLflow tracking URI
# os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
# os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

# # Load the latest version of the model from the registry
# model_name = "SentimentAnalysisNLP"
# model_version = "6"
# # stage = "Production"  # or "Staging", "None", etc.

# model_version = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# model = model_version._model_impl.model
# tokenizer = BertTokenizer.from_pretrained(model_dir)

# # Load dataset
# data = pd.read_csv('sentiment_analysis/datasets/train_data.csv')
# texts = data['text'].tolist()
# labels = data['label'].tolist()

# # Prepare inputs
# inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
# labels = torch.tensor(labels)

# # Fine-tune model
# model.train()
# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# loss.backward()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# optimizer.step()

# # Log the retrained model back to the MLflow Model Registry
# with mlflow.start_run(run_name="retrained_sentiment_model"):
#     mlflow.pyfunc.log_model(
#         artifact_path="model",
#         python_model=SentimentAnalysisModel(),
#         registered_model_name=model_name
#     )
#     mlflow.log_metric("training_loss", loss.item())
#     mlflow.log_param("learning_rate", 1e-5)

#     # Optionally promote the new model to a particular stage (e.g., "Production")
#     model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
#     mlflow.register_model(model_uri=model_uri, name=model_name)

# print("Model retrained and logged successfully.")
import os
import mlflow
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# Set up MLflow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

# Model parameters
model_name = "SentimentAnalysisNLP"
model_version = "10"

# Load model from MLflow model registry
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

# Load dataset
data = pd.read_csv('sentiment_analysis/datasets/train_data.csv')
texts = data['text'].tolist()
labels = data['label'].tolist()

# Prepare inputs using the tokenizer from the loaded model
tokenizer = model._model_impl.tokenizer
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
labels = torch.tensor(labels)

# Fine-tune model
model._model_impl.model.train()
outputs = model._model_impl.model(**inputs, labels=labels)
loss = outputs.loss
loss.backward()
optimizer = torch.optim.Adam(model._model_impl.model.parameters(), lr=1e-5)
optimizer.step()

# Log the retrained model
with mlflow.start_run(run_name="retrained_sentiment_model"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=SentimentAnalysisModel(),
        registered_model_name=model_name
    )
    mlflow.log_metric("training_loss", loss.item())
    mlflow.log_param("learning_rate", 1e-5)

print("Model retrained and logged successfully.")

