#     # Register the model in the MLflow Model Registry
#     model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
#     mlflow.register_model(model_uri=model_uri, name="SentimentAnalysisNLP")

import os
import mlflow
import mlflow.pyfunc
import requests
import pandas as pd
import torch
from mlflow.models.signature import infer_signature

# Setup MLflow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

files = {
    "https://dagshub.com/valiant.shabri/dagster/src/main/sentiment_analysis/models/config.json": "/app/models/config.json",
    "https://dagshub.com/valiant.shabri/dagster/src/main/sentiment_analysis/models/model.safetensors": "/app/models/model.safetensors",
    "https://dagshub.com/valiant.shabri/dagster/src/main/sentiment_analysis/models/vocab.txt": "/app/models/vocab.txt",
    "https://dagshub.com/valiant.shabri/dagster/src/main/sentiment_analysis/models/tokenizer_config.json": "/app/models/tokenizer_config.json",
    "https://dagshub.com/valiant.shabri/dagster/src/main/sentiment_analysis/models/special_tokens_map.json": "/app/models/special_tokens_map.json"
}

for url, path in files.items():
    if not os.path.exists(path):
        response = requests.get(url)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(response.content)


# Define model directory and model name
model_dir = "/app/models"  # Path to the directory containing your model files
registered_model_name = "SentimentAnalysisNLP"
input_example = ["semuanya masih proses awal belum masuk dalam rapat dpp dan belum dilaporkan kepada ibu ketua umum"]

from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification

tokenizer = BertTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Example input for signature inference
inputs = tokenizer(input_example, return_tensors="pt", padding=True)
signature = infer_signature(pd.DataFrame({"text": input_example}), model(**inputs).logits.detach().numpy())

# Define a custom model class to log
class SentimentAnalysisModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        print("Available Artifacts: ", context.artifacts)
        model_path = context.artifacts["model_dir"]
        print(f"Loading model from: {model_path}")
        print(f"Files in model directory: {os.listdir(model_path)}")

        try:
            # Ensure the correct path is used
            self.tokenizer = BertTokenizer.from_pretrained(context.artifacts["model_dir"])
            self.model = AutoModelForSequenceClassification.from_pretrained(context.artifacts["model_dir"])
            print("Model and tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def predict(self, context, model_input):
        # Check if model_input is a DataFrame, convert it to a list if it is
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.iloc[:, 0].tolist()  # Convert the first column to list
        elif isinstance(model_input, list):
            pass  # Already a list, no need to change


        inputs = self.tokenizer(model_input.tolist(), return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        return outputs.logits.argmax(dim=1).numpy()

# Initialize an MLflow run
with mlflow.start_run(run_name="Sentiment_Analysis_Model_Log"):
    # Log the custom model
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=SentimentAnalysisModel(),
        registered_model_name=registered_model_name,
        input_example=input_example,
        signature=signature,
        artifacts={"model_dir": model_dir}  # Ensure the correct path is passed as an artifact
    )

    tokenizer_dir = os.path.join(model_dir, "tokenizer")
    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    
    # Optionally log other artifacts or parameters
    mlflow.log_artifacts(tokenizer_dir, artifact_path="tokenizer")
    mlflow.log_param("model_type", "ALBERT")
    mlflow.log_param("tokenizer", "BERT")

print("Model logged and registered successfully.")