import os
import mlflow
import mlflow.pyfunc
import mlflow
from mlflow.models.signature import infer_signature

# Setup MLflow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

model_dir = "/models"  # Path to the directory containing your model files

# Define a custom model class to log
class SentimentAnalysisModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification
        
        # Load the model components
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)

    def predict(self, context, model_input):
        inputs = self.tokenizer(model_input.tolist(), return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        return outputs.logits.argmax(dim=1).numpy()

# Initialize an MLflow run
with mlflow.start_run(run_name="Sentiment_Analysis_Model_Log"):
    # Log the custom model
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=SentimentAnalysisModel(),
        registered_model_name="SentimentAnalysisNLP"
    )
    
    # Optionally log other artifacts or parameters
    mlflow.log_artifacts(model_dir)
    mlflow.log_param("model_type", "BERT")

print("Model logged successfully.")
