from typing import List, Dict, Tuple
import os
import joblib
import torch
import logging
from src.data_loader import DataLoader
from src.adfler import MessageSegmenter
from src.lasec import LogAbstractor
from src.utils import get_latest_folder
import json
from dataclasses import asdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DroneLogAnalyzer:
    def __init__(self, config: Dict):
        # Initialize all components from config
        self.config = config
        self.data_loader = DataLoader(config['data_path'])
        self.segmenter = MessageSegmenter(config['ner_model_path'], use_cuda=config['use_cuda'])
        self.abstractor = self._load_lasec()
        # self.detector = AnomalyDetector(
        #     config['severity_model_path'],
        #     config['classifier_path']
        # )
        # self.attributor = AttributionAnalyzer(
        #     config['attribution_model_path'],
        #     config['classifier_path']
        # )
        # self.seq_db_builder = SequenceBuilder()

    def _load_lasec(self):
        file_path = os.path.join(self.config['workdir'], 'LASeC.joblib')

        if os.path.exists(file_path):
            self.abstractor = joblib.load(file_path)
        else:
            self.abstractor = LogAbstractor(
                self.config['embedding_model_path'],
                self.config['device'],
                self.config.get('birch_model_path')
            )

    def analyze(self):
        """Run the complete analysis pipeline"""
        # 1. Load data
        logger.info(f'Load data from the file...')
        records = self.data_loader.load_data()
        
        # 2. Segment messages
        logger.info(f'Start segmenting messages...')
        records = self.segmenter.segment_and_classify(records)
        
        # # 3. Abstract events
        records = self.abstractor.abstract_messages(records)
        joblib.dump(self.abstractor, os.path.join(self.config['workdir'], 'LASeC.joblib'))

        # # 4. Detect anomalies
        # records = self.detector.detect_anomalies(records)
        
        # # 5. Compute attributions
        # records = self.attributor.compute_attributions(records)
        
        # # 6. Build sequences
        # result = self.builder.build_sequences(records)
        
        return records
        return result


def main():
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parsed_folder = get_latest_folder('outputs')
    source_path = os.path.join('outputs', parsed_folder, 'parsed', 'android')
    files = os.listdir(source_path)
    output_dir = os.path.join('outputs', parsed_folder, 'records')
    os.makedirs(output_dir, exist_ok=True)

    for file in files:
        full_path = os.path.join(source_path, file)
        logger.info(f'Processing file: {full_path}')

        config = {
            'work_dir': parsed_folder,
            'use_cuda': use_cuda,
            'device': device,
            'data_path': full_path,
            'ner_model_path': 'ADFLER-albert-base-v2',
            'embedding_model_path': 'swardiantara/drone-sbert',
            'birch_model_path': os.path.join(parsed_folder, 'birch_model.joblib'),
            # 'severity_model_path': 'severity_model',
            # 'classifier_path': 'classifier.pt',
            # 'attribution_model_path': 'attribution_model'
        }
        
        analyzer = DroneLogAnalyzer(config)
        results = analyzer.analyze()
        
        # Convert LogRecord objects to dictionaries
        serializable_results = [asdict(record) for record in results]
        # Save or process results
        with open(os.path.join(output_dir, f'{file}.json'), 'w') as f:
            json.dump(serializable_results, f, indent=2)


# Example usage
if __name__ == "__main__":
    main()