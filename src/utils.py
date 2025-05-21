import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from torch.utils.data import Dataset

from src.data_loader import LogRecord

def get_latest_folder(directory='.'):
    # Get all folders in the directory
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    # Filter folders matching the pattern YYYYMMDD_HHmmss
    pattern = re.compile(r'^\d{8}_\d{6}$')
    matching_folders = [f for f in folders if pattern.match(f)]
    
    # Sort by name (which effectively sorts by date and time given the format)
    matching_folders.sort()
    
    if matching_folders:
        return os.path.join(directory, matching_folders[-1])
    else:
        return None
    

class AnomalyDataset(Dataset):
    def __init__(self, sentences: List[str], tokenizer, max_length=64):
        self.data = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    

def get_device(model):
    for param in model.parameters():
        return param.device
    for buffer in model.buffers():
        return buffer.device
    raise ValueError("Model has no parameters or buffers on any device.")