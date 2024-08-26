import os
import mlflow
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

# Load test dataset
test_data = pd.read_csv('datasets/test_data.csv')
texts = test_data['text'].tolist()
true_labels = test_data['label'].tolist()

# Load model
model_uri = f"models:/SentimentAnalysisNLP/latest"
model = mlflow.pyfunc.load_model(model_uri)
tokenizer = BertTokenizer.from_pretrained('model')

# Prepare inputs
inputs = tokenizer(texts, return_tensors="pt", padding=True)
preds = model.predict(inputs)

# Calculate accuracy
accuracy = (preds == true_labels).mean()
print(f"Test Accuracy: {accuracy}")
