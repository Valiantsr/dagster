import os
import mlflow
import torch
from transformers import AutoTokenizer
from torch.nn.functional import softmax
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Set up MLflow tracking
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

# Load the tokenizer and model from MLflow model registry
model_uri = "models:/IndoBERT_Sentiment_Model/latest"
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = mlflow.pytorch.load_model(model_uri)
model.eval()  # Set the model to evaluation mode

# Define label mapping for the sentiment classification
label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

# Define request body model
class SentimentRequest(BaseModel):
    text: str

# Define route for sentiment prediction
@app.post("/predict/")
async def predict_sentiment(request: SentimentRequest):
    try:
        # Tokenize input text
        inputs = tokenizer(request.text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

        # Make prediction using the model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()

        # Get label and probability
        predicted_label = label_mapping[prediction]
        predicted_prob = probabilities[0][prediction].item()

        return {
            "text": request.text,
            "predicted_label": predicted_label,
            "predicted_probability": predicted_prob
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define a root route for health check
@app.get("/")
def read_root():
    return {"message": "IndoBERT Sentiment Analysis API is running."}

# Main entry point to run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)