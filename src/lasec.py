import os
import joblib
from src.data_loader import LogRecord
from typing import List, Dict, Tuple, DefaultDict
from collections import defaultdict
from sklearn.cluster import Birch
from sentence_transformers import SentenceTransformer


class LogAbstractor:
    def __init__(self, model_path: str, device, birch_model_path: str = None):
        self.birch_model_path = birch_model_path
        self.embedding_model = SentenceTransformer(model_path, device=device)
        self.birch_model = self._load_birch_model(birch_model_path)
        self.cluster_members: DefaultDict[int, List[str]] = defaultdict(list)
        
    def _load_birch_model(self, path):
        if os.path.exists(path):
            # Load pre-trained BIRCH model
            return joblib.load(path)
        else:
            # Initialize new BIRCH model
            return Birch(n_clusters=None, threshold=0.3)
        
    def _save_birch_model(self):
        joblib.dump(self.birch_model, self.birch_model_path)
    
    def abstract_messages(self, records: List[LogRecord]) -> List[LogRecord]:
        """Assign event IDs to each sentence via semantic clustering"""
        # Collect all sentences for batch processing
        all_sentences = []
        sentence_refs = []  # To map back to original records
        
        for i, record in enumerate(records):
            for j, sentence in enumerate(record.sentences):
                all_sentences.append(sentence)
                sentence_refs.append((i, j))
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(all_sentences, normalize_embeddings=True)
        
        # Cluster (or predict if pre-trained)
        if self.birch_model.n_clusters is None:
            clusters = self.birch_model.fit_predict(embeddings)
        else:
            clusters = self.birch_model.predict(embeddings)
        
        # Assign event IDs back to records
        for (i, j), cluster_id in zip(sentence_refs, clusters):
            sentence_type = records[i].sentence_types[j]
            prefix = sentence_type[0] #'E' if sentence_type == 'event' else 'N'
            event_id = f"{prefix}{cluster_id}"
            
            records[i].eventIds.append(event_id)
            self.cluster_members[cluster_id].append(records[i].sentences[j])
        # Save the last state for online setting
        self._save_birch_model()

        return records