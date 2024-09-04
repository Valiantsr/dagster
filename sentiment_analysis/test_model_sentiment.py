import os
import mlflow
import torch
from transformers import AutoTokenizer
import pandas as pd
import requests

# Konfigurasi MLflow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

# Kalimat untuk testing
sentences = [
    "polri bertugas secara profesional dan tak pandang bulu.",
    "semuanya masih proses awal belum masuk dalam rapat dpp dan belum dilaporkan kepada ibu ketua umum",
    "enggak ada ancaman, cuman dibilangnya percuma punya teman punya saudara jadi pj gubernur, tapi gak ada gunanya"
]

# Load model dari MLflow Model Registry
model_uri = f"models:/SentimentAnalysisNLP/latest"
loaded_model = mlflow.pytorch.load_model(model_uri)

# Load tokenizer yang sesuai dengan model yang diload
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

# Tokenisasi input sentences
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# Gunakan model untuk prediksi
with torch.no_grad():
    outputs = loaded_model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

# Konversi tensor prediksi menjadi list
predicted_labels = predictions.tolist()

# Output hasil prediksi
for sentence, label in zip(sentences, predicted_labels):
    print(f"Sentence: {sentence}")
    print(f"Predicted Label: {label}\n")