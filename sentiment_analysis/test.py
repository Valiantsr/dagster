import os
import mlflow
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# Set up MLflow tracking
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

# Load the latest model from MLflow
model_uri = f"models:/IndoBERT_Sentiment_Model/latest"
model = mlflow.pytorch.load_model(model_uri)

# Custom sentences to test
sentences = [
    "polri bertugas secara profesional dan tak pandang bulu.",
    "semuanya masih proses awal belum masuk dalam rapat dpp dan belum dilaporkan kepada ibu ketua umum",
    "enggak ada ancaman, cuman dibilangnya percuma punya teman punya saudara jadi pj gubernur, tapi gak ada gunanya"
]

# Tokenize the sentences
inputs = tokenizer(sentences, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

# Model evaluation
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    probabilities = softmax(outputs.logits, dim=-1)

# Map the predictions back to the label names
label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
predicted_labels = [label_mapping[pred.item()] for pred in predictions]
predicted_probs = probabilities.tolist()

# Log results with MLflow
with mlflow.start_run(run_name="IndoBERT_Sentiment_Custom_Testing"):
    for i, sentence in enumerate(sentences):
        print(f"Sentence: {sentence}")
        print(f"Predicted Label: {predicted_labels[i]} ({predicted_probs[i]})")
        mlflow.log_metric(f"sentence_{i}_predicted_label", predictions[i].item())
        mlflow.log_metric(f"sentence_{i}_predicted_prob", max(predicted_probs[i]))

        mlflow.log_param(f"sentence_{i}_text", sentence)

print("Testing complete.")