import os
import mlflow
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import pandas as pd
from sklearn.metrics import accuracy_score
from torch.nn.functional import softmax

# Set up MLflow tracking
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/valiant.shabri/dagster.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'valiant.shabri'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd37b33ad4e0564f52162d90248e477d373a699f1'

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs

# Load datasets
def load_dataset(data_url):
    data = pd.read_csv(data_url)
    texts = data['text'].tolist()
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    labels = [label_mapping[label] for label in data['label'].tolist()]
    return texts, labels

train_texts, train_labels = load_dataset("https://dagshub.com/api/v1/repos/valiant.shabri/dagster/storage/raw/s3/dagster/data/train.csv")
valid_texts, valid_labels = load_dataset("https://dagshub.com/api/v1/repos/valiant.shabri/dagster/storage/raw/s3/dagster/data/validate.csv")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

# Create DataLoaders
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
valid_dataset = SentimentDataset(valid_texts, valid_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)

# Load the latest model from the registry
model_name = "IndoBERT_Sentiment_Model"
client = mlflow.tracking.MlflowClient()
latest_version = client.get_latest_versions(model_name, stages=["None"])[-1].version
model_uri = f"models:/{model_name}/{latest_version}"
model = mlflow.pytorch.load_model(model_uri)

# Training and validation function
def train_and_validate(model, train_loader, valid_loader, epochs=3, learning_rate=2e-5):
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_train_loss, total_train_accuracy = 0, 0
        total_valid_loss, total_valid_accuracy = 0, 0

        # Training loop
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            accuracy = accuracy_score(batch['labels'].cpu(), predictions.cpu())
            total_train_accuracy += accuracy

        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                outputs = model(**batch)
                loss = outputs.loss
                total_valid_loss += loss.item()

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                accuracy = accuracy_score(batch['labels'].cpu(), predictions.cpu())
                total_valid_accuracy += accuracy

        # Log metrics to MLflow
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_accuracy = total_train_accuracy / len(train_loader)
        avg_valid_loss = total_valid_loss / len(valid_loader)
        avg_valid_accuracy = total_valid_accuracy / len(valid_loader)

        mlflow.log_metric(f"train_loss_epoch_{epoch+1}", avg_train_loss)
        mlflow.log_metric(f"train_accuracy_epoch_{epoch+1}", avg_train_accuracy)
        mlflow.log_metric(f"valid_loss_epoch_{epoch+1}", avg_valid_loss)
        mlflow.log_metric(f"valid_accuracy_epoch_{epoch+1}", avg_valid_accuracy)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {avg_train_accuracy:.4f}")
        print(f"Valid Loss: {avg_valid_loss:.4f} | Valid Accuracy: {avg_valid_accuracy:.4f}")

if __name__ == "__main__":
    mlflow.pytorch.autolog()

    with mlflow.start_run(run_name="IndoBERT_Sentiment_Retrain") as run:
        # Log hyperparameters
        mlflow.log_param("epochs", 3)
        mlflow.log_param("learning_rate", 2e-5)
        mlflow.log_param("batch_size", 16)
        mlflow.log_param("model_name", "indobenchmark/indobert-base-p1")

        train_and_validate(model, train_loader, valid_loader, epochs=3, learning_rate=2e-5)
        
        # Log the retrained model
        mlflow.pytorch.log_model(model, "model")
        
        latest_version = int(latest_version)  # Convert the latest version to an integer
        new_version = latest_version + 1  # Increment the version number

        # Register the retrained model to MLflow Model Registry
        mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)

        # Optionally, promote the retrained model to 'Production' stage
        client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage="Production"
        )
