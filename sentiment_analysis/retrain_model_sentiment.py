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

class SentimentAnalysisModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = context.artifacts["model_dir"]
        config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def predict(self, context, model_input):
        inputs = self.tokenizer(model_input["text"].tolist(), return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits.argmax(dim=1).numpy()

# model_dir = loaded_model._model_impl.artifacts["model_dir"]
# model = mlflow.pyfunc.load_model(model_uri)

# Access the tokenizer and model from the context
# model_dir = loaded_model._model_impl.artifacts["model_dir"]
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForSequenceClassification.from_pretrained(model_dir)
# tokenizer = model._model_impl.tokenizer
# model = model._model_impl.model

url = "https://dagshub.com/api/v1/repos/valiant.shabri/dagster/storage/raw/s3/dagster/data/train.tsv"
local_path = 'sentiment_analysis/datasets/test.tsv'

# Jika file belum ada di direktori lokal, unduh dari DagsHub
if not os.path.exists(local_path):
    response = requests.get(url)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, 'wb') as f:
        f.write(response.content)


# Load dataset
data = pd.read_csv('sentiment_analysis/datasets/train_data.tsv', sep='\t')
texts = data['text'].tolist()
labels = data['label'].tolist()

# Prepare inputs using the tokenizer
# inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
inputs = loaded_model._model_impl.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
labels = torch.tensor(labels)

# Fine-tune model
model = loaded_model._model_impl.model
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
        # python_model=model._model_impl,
        python_model=SentimentAnalysisModel(),  # Use the model from the registry
        registered_model_name=registered_model_name,
        artifacts={"model_dir": loaded_model._model_impl.artifacts["model_dir"]}  # Correct path for saving artifacts
    )
    mlflow.log_metric("training_loss", loss.item())
    mlflow.log_param("learning_rate", 1e-5)

print("Model retrained and logged successfully.")
