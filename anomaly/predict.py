import os
import pandas as pd

import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

from utils import PredictDataset
from model import AnomalySeverity
from train import raw2label, label2idx, idx2label

def predict(dataset: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name_path = "swardiantara/drone-ordinal-all"
    tokenizer = AutoTokenizer.from_pretrained(model_name_path)
    embedding_model = AutoModel.from_pretrained(model_name_path).to(device)

    # Define the custom dataset and dataloaders
    max_seq_length = 64
    batch_size = 64

    test_dataset = PredictDataset(dataset, tokenizer, max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AnomalySeverity(embedding_model, tokenizer).to(device)
    model.load_state_dict(torch.load('pytorch_model.pt'))
    predictions = []
    model.eval()
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1)
        predictions.extend(predicted_class)
    dataset['anomaly'] = predictions
    dataset['anomaly'] = dataset['anomaly'].map(idx2label)
    dataset.to_excel(os.path.join(output_dir, 'anomaly_severity.xlsx'))

    return exit(0)


if __name__ == "__main__":
    test_data = pd.read_excel(os.path.join('..', 'data', 'parsed_DJIFlightRecord_2025-05-12_[08-20-56].csv.xlsx'))
    predict(test_data, 'output')