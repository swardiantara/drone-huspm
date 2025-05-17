import os
import copy
import random
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

from utils import BaselineDataset
from model import AnomalySeverity

raw2label = {
        1: 'normal',
        2: 'low',
        3: 'medium',
        4: 'high'
    }

label2idx = {
    'normal': 0,
    'low': 1,
    'medium': 2,
    'high': 3
}

idx2label = {
    0: 'normal',
    1: 'low',
    2: 'medium',
    3: 'high'
}

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


def main():
    # Set global seed for reproducibility
    set_seed()
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = pd.read_csv('dataset.csv')
    dataset["label"] = dataset['label'].map(raw2label)
    dataset["multiclass_label"] = dataset['label'].map(label2idx)

    model_name_path = "swardiantara/drone-ordinal-all"
    tokenizer = AutoTokenizer.from_pretrained(model_name_path)
    embedding_model = AutoModel.from_pretrained(model_name_path).to(device)

    # Define the custom dataset and dataloaders
    max_seq_length = 64
    batch_size = 64
    num_epochs = 10

    train_dataset = BaselineDataset(dataset, tokenizer, max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = AnomalySeverity(embedding_model, tokenizer).to(device)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_multiclass_train = batch["labels_multiclass"].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            loss_multiclass_train = criterion(logits, labels_multiclass_train)
            loss_multiclass_train.backward()
            optimizer.step()

            total_train_loss += loss_multiclass_train.item()
        train_loss_epoch = total_train_loss / len(train_loader)
        print(f"{epoch+1}/{num_epochs}: train_loss: {train_loss_epoch}/{total_train_loss}")
    best_model_state = copy.deepcopy(model.state_dict())
    # Save the model
    torch.save(best_model_state, 'pytorch_model.pt')

    return exit(0)


if __name__ == "__main__":
    main()