import torch
import numpy as np
import torch.nn as nn
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from captum.attr import LayerIntegratedGradients

from src.data_loader import LogRecord
from src.utils import get_device

idx2label = {
    0: 'normal',
    1: 'low',
    2: 'medium',
    3: 'high'
}

label2idx = {
    'normal': 0,
    'low': 1,
    'medium': 2,
    'high': 3
}

class AnomalyDetector(nn.Module):
    def __init__(self, embedding_model, tokenizer, hidden_dim=128, dropout_rate=0.1, num_class=4, freeze_embedding=False):
        """
        Args:
            embedding_model: A Hugging Face-compatible transformer model with a .forward method
            tokenizer: The corresponding tokenizer
            hidden_dim: Size of the hidden layer in the classifier
            dropout_rate: Dropout probability
            num_class: Number of classes in the output
            freeze_embedding: Whether to freeze the embedding's parameters
        """
        super(AnomalyDetector, self).__init__()
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer

        if hasattr(embedding_model.config, 'hidden_size'):
            self.embedding_dim = embedding_model.config.hidden_size
        else:
            raise ValueError("Could not determine embedding dimension from model config.")
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_class)
        )

        if freeze_embedding:
            for param in self.embedding_model.parameters():
                param.requires_grad = False

    def mean_pooling(self, last_hidden_state, attention_mask):
        # attention_mask: [batch_size, seq_len]
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [batch_size, seq_len, 1]
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts  # [batch_size, hidden_size]

    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        pooled = self.mean_pooling(last_hidden_state, attention_mask)  # [batch_size, hidden_dim]
        logits = self.classifier(pooled)  # [batch_size, num_classes]
        return logits
    
    def signal_to_noise_ratio(self, attribution_array):
        # Get only positive attributions
        positive_attributions = np.maximum(attribution_array, 0)
        
        # Sum of positive attributions (signal)
        signal = np.sum(positive_attributions)
        
        # Standard deviation of all attributions (noise)
        noise = np.std(attribution_array) + 1e-10  # epsilon to prevent division by zero
        
        # Calculate SNR
        snr = signal / noise
        
        return snr
    
    def compute_attribution(self, input_ids, attention_mask):
        lig = LayerIntegratedGradients(self, self.embedding_model.embeddings)
        target_class = label2idx.get('high') # Example target class
        attributions, delta = lig.attribute(inputs=input_ids, 
                                            baselines=input_ids*0, 
                                            additional_forward_args=(attention_mask,),
                                            target=target_class,
                                            return_convergence_delta=True)
        # Sum the attributions across embedding dimensions
        attributions = attributions.sum(dim=-1).squeeze(0)
        # Normalize the attributions for better visualization
        attributions = attributions / torch.norm(attributions)
        # Convert attributions to numpy
        attributions = attributions.cpu().detach().numpy()
        snr = self.signal_to_noise_ratio(attributions)
        return snr

    def detect_anomalies(self, records: List[LogRecord]) -> List[LogRecord]:
        """Classify severity for each abstracted event"""
        device = get_device(self)
        self.eval()
        # prepare the list of sentences
        for record in records:
            if not record.sentences:
                continue
                
            for sentence in record.sentences:
                with torch.no_grad():
                    inputs = self.tokenizer(sentence, padding=True, truncation=True, return_tensors="pt").to(device)
                    logits = self(inputs["input_ids"], inputs["attention_mask"])
                    pred_prob = torch.softmax(logits, dim=-1)
                    pred_label = torch.argmax(pred_prob, dim=-1).item()
                    prob = pred_prob[0, pred_label].item()
                    attribution = self.compute_attribution(inputs["input_ids"], inputs["attention_mask"])
                    record.anomalies.append(idx2label.get(pred_label))
                    record.anomaly_probs.append(prob)
                    record.attributions.append(attribution)
        
        return records
