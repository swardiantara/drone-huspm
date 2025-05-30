import os
import numpy as np
import joblib
import json
from src.data_loader import LogRecord
from typing import List, Dict, Tuple, DefaultDict
from collections import defaultdict
from sklearn.cluster import Birch, AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from sentence_transformers import SentenceTransformer


class LogAbstractor:
    def __init__(self, device, model_path: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_path, device=device)
        self.cluster_model = self._get_cluster_model()
        self.sentence_cluster_members: DefaultDict[int, list[str]] = defaultdict(list)
        self.message_cluster_members: DefaultDict[int, list[str]] = defaultdict(list)
        self.sentence_representative_log: DefaultDict[str, str] = defaultdict()
        self.message_representative_log: DefaultDict[str, str] = defaultdict()
        self.sentence_problem: DefaultDict[str, DefaultDict[str, str]] = self._construct_problem()
        self.message_problem: DefaultDict[str, DefaultDict[str, str]] = self._construct_problem()
        
    def _get_cluster_model(self, threshold: float = 0.2, linkage: str = 'average'):
        return AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=threshold,
                    linkage=linkage,
                    metric='precomputed')
    
    def _construct_problem(self):
        return {
            'multiclass': defaultdict(),
            'binary': defaultdict()
        }
    
    def compute_distance_matrix(self, corpus_embeddings, metric: str = 'cosine', is_norm=False):
        if is_norm:
            corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

        distance_matrix = pairwise_distances(corpus_embeddings, corpus_embeddings, metric=metric)
        return distance_matrix
        
    def save_cluster_member(self, sentence_path, message_path):
        cleaned_cluster_members = {
            int(cid): list(set(members))
            for cid, members in self.sentence_cluster_members.items()
        }
        with open(sentence_path, 'w') as f:
            json.dump(cleaned_cluster_members, f, indent=2)
        cleaned_cluster_members = {
            int(cid): list(set(members))
            for cid, members in self.message_cluster_members.items()
        }
        with open(message_path, 'w') as f:
            json.dump(cleaned_cluster_members, f, indent=2)

    def save_representative_log(self, sentence_path, message_path):
        with open(sentence_path, 'w') as f:
            json.dump(self.sentence_representative_log, f, indent=2)
        with open(message_path, 'w') as f:
            json.dump(self.message_representative_log, f, indent=2)

    def save_problem(self, sentence_path, message_path):
        with open(sentence_path, 'w') as f:
            json.dump(self.sentence_problem, f, indent=2)
        with open(message_path, 'w') as f:
            json.dump(self.message_problem, f, indent=2)

    def abstract_messages(self, records: List[LogRecord]) -> List[LogRecord]:
        """Assign event IDs to each sentence via semantic clustering"""
        # Collect all sentences for batch processing
        all_messages = []
        all_sentences = []
        sentence_refs = []  # To map back to original records
        
        for i, record in enumerate(records):
            for j, sentence in enumerate(record.sentences):
                # if (record.sentence_types[j] == 'Event') and (not record.anomalies[j] == 'normal'):
                all_messages.append(record.raw_message)
                all_sentences.append(sentence)
                sentence_refs.append((i, j))

        # Generate message embeddings
        message_embeddings = self.embedding_model.encode(all_messages, normalize_embeddings=True)
        message_distance_matrix = self.compute_distance_matrix(message_embeddings)
        message_clusters = self.cluster_model.fit_predict(message_distance_matrix)
        # Assign event IDs back to records
        for i, cluster_id in enumerate(message_clusters):
            event_id = f"E{cluster_id}"
            records[i].message_eventId = event_id
            self.message_cluster_members[cluster_id].append(records[i].raw_message)

            # store representative log
            if not cluster_id in self.message_representative_log:
                indices = np.where(message_clusters == cluster_id)[0]
                cluster_embeds = message_embeddings[indices]
                centroid = np.mean(cluster_embeds, axis=0)
                distances = np.linalg.norm(cluster_embeds - centroid, axis=1)
                self.message_representative_log[event_id] = all_messages[indices[np.argmin(distances)]]

                if records[i].message_anomaly != 'normal':
                    problem_id = f'{event_id}-{records[i].message_anomaly}'
                    self.message_problem['multiclass'][problem_id] = self.message_representative_log[event_id]
                    if not event_id in self.message_problem['binary']:
                        self.message_problem['binary'][event_id] = self.message_representative_log[event_id]
        
        # Generate sentence embeddings
        sentence_embeddings = self.embedding_model.encode(all_sentences, normalize_embeddings=True)
        sentence_distance_matrix = self.compute_distance_matrix(sentence_embeddings)
        sentence_clusters = self.cluster_model.fit_predict(sentence_distance_matrix)
        
        # Assign event IDs back to records
        for (i, j), cluster_id in zip(sentence_refs, sentence_clusters):
            sentence_type = records[i].sentence_types[j]
            anomaly = records[i].anomalies[j]
            event_id = f"{sentence_type[0]}{cluster_id}" #'E' if sentence_type == 'event' else 'N'
            
            records[i].eventIds.append(event_id)
            self.sentence_cluster_members[cluster_id].append(records[i].sentences[j])

            # store representative log
            if not cluster_id in self.sentence_representative_log:
                indices = np.where(sentence_clusters == cluster_id)[0]
                cluster_embeds = sentence_embeddings[indices]
                centroid = np.mean(cluster_embeds, axis=0)
                distances = np.linalg.norm(cluster_embeds - centroid, axis=1)
                self.sentence_representative_log[event_id] = all_sentences[indices[np.argmin(distances)]]

                if sentence_type == 'Event' and anomaly != 'normal':
                    problem_id = f'{event_id}-{anomaly}'
                    self.sentence_problem['multiclass'][problem_id] = self.sentence_representative_log[event_id]
                    if not event_id in self.sentence_problem['binary']:
                        self.sentence_problem['binary'][event_id] = self.sentence_representative_log[event_id]
            
        return records