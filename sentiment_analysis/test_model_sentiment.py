import os
import mlflow
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import requests

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

# URL untuk mengunduh test_data.csv dari DagsHub
url = "https://dagshub.com/valiant.shabri/dagster/src/main/s3:/dagster/data/test.csv"
local_path = 'sentiment_analysis/datasets/test.csv'

# Jika file belum ada di direktori lokal, unduh dari DagsHub
if not os.path.exists(local_path):
    response = requests.get(url)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, 'wb') as f:
        f.write(response.content)

# Load test dataset
test_data = pd.read_csv(local_path, on_bad_lines='skip')
texts = test_data['text'].tolist()
true_labels = test_data['label'].tolist()

# Load model
model_uri = f"models:/SentimentAnalysisNLP/latest"
model = mlflow.pyfunc.load_model(model_uri)
# tokenizer = BertTokenizer.from_pretrained('model')

# Prepare inputs
# inputs = tokenizer(texts, return_tensors="pt", padding=True)
# preds = model.predict(inputs)
inputs = model._model_impl.tokenizer(texts, return_tensors="pt", padding=True)
preds = model.predict(pd.DataFrame({'text': texts}))

# Calculate accuracy
accuracy = (preds == true_labels).mean()
print(f"Test Accuracy: {accuracy}")
