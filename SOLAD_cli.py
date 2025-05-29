from typing import List, Dict, Tuple
import os
import joblib
import torch
import logging
import json
from dataclasses import asdict

from transformers import AutoModel, AutoTokenizer

from src.data_loader import DataLoader
from src.adfler import MessageSegmenter
from src.lasec import LogAbstractor
from src.dronelog import AnomalyDetector
from src.seq_builder import SequenceBuilder
from src.report_generator import ReportGenerator
from src.utils import get_latest_folder

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("simpletransformers.ner.ner_model").setLevel(logging.WARNING)

class DroneLogAnalyzer:
    def __init__(self, config: Dict):
        # Initialize all components from config
        self.config = config
        self.data_loader = DataLoader(config['data_path'])
        self.segmenter = MessageSegmenter(config['ner_model_path'], use_cuda=config['use_cuda'])
        self.abstractor = self._load_lasec()
        self.detector = self._load_dronelog()
        self.report_generator = ReportGenerator(config)
        # self.attributor = AttributionAnalyzer(
        #     config['attribution_model_path'],
        #     config['classifier_path']
        # )
        # self.seq_db_builder = SequenceBuilder()

    def _load_lasec(self):
        return LogAbstractor(
            self.config['device'],
            self.config['embedding_model_path']
        )
        
    def _load_dronelog(self):
        pre_trained = self.config['classifier_path']
        if not os.path.exists(pre_trained):
            raise NotImplementedError('The anomaly severity detection model is not found!')
        
        tokenizer = AutoTokenizer.from_pretrained(self.config['severity_model_path'])
        embedding_model = AutoModel.from_pretrained(self.config['severity_model_path']).to(self.config['device'])

        model = AnomalyDetector(embedding_model, tokenizer).to(self.config['device'])
        model.load_state_dict(torch.load(pre_trained, map_location=self.config['device']))
        return model
    
    def analyze(self):
        """Run the complete analysis pipeline"""
        # 1. Load data
        logger.info(f'Load data from the file...')
        records = self.data_loader.load_data()
        logger.info(f'Data successfully loaded!')
        
        # 2. Segment messages
        logger.info(f'Start event recognition...')
        # records = self.segmenter.segment_and_classify(records)
        records = self.segmenter.syntactic_segmenter(records)
        logger.info(f'Event recognition completed successfully!')
        
        # 3. Detect anomalies
        logger.info(f'Start anomaly severity detection...')
        records = self.detector.detect_anomalies(records)
        logger.info(f'Anomaly severity detection completed successfully!')

        # 4. Abstract events
        logger.info(f'Start event abstraction...')
        records = self.abstractor.abstract_messages(records)
        output_dir = os.path.join(self.config['workdir'], 'cluster')
        os.makedirs(output_dir, exist_ok=True)
        self.abstractor.save_cluster_member(os.path.join(output_dir, f'{self.config['filename'].split('.')[0]}_cluster_mapping.json'))
        output_dir = os.path.join(self.config['workdir'], 'event')
        os.makedirs(output_dir, exist_ok=True)
        self.abstractor.save_representative_log(os.path.join(output_dir, f'{self.config['filename'].split('.')[0]}_event.json'))
        output_dir = os.path.join(self.config['workdir'], 'problem')
        os.makedirs(output_dir, exist_ok=True)
        self.abstractor.save_problem(os.path.join(output_dir, f'{self.config['filename'].split('.')[0]}_problem.json'))
        logger.info(f'Event abstraction completed successfully!')

        # # 5. Report Generation
        output_dir = os.path.join(self.config['workdir'], 'report')
        os.makedirs(output_dir, exist_ok=True)
        self.report_generator.create_timeline_chart(records, self.abstractor.problem['binary'], output_dir)
        # records = self.attributor.compute_attributions(records)
        
        # 6. Build sequences per log file, and save to workdir
        # logger.info(f'Start constructing sequence DB...')
        # for utility in self.config['utility_variable']:
        #     seq_dir = os.path.join(self.config['workdir'], 'sequence', utility)
        #     os.makedirs(seq_dir, exist_ok=True)
        #     result = self.seq_db_builder.build_sequences(records, utility)
        #     joblib.dump(result, os.path.join(seq_dir, f'{self.config['filename'].split('.')[0]}_sequence.joblib'))
        
        return records


def main():
    use_cuda = True if torch.cuda.is_available() else False
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    parsed_folder = get_latest_folder('outputs')
    source_path = os.path.join(parsed_folder, 'parsed', 'android')
    files = os.listdir(source_path)

    for file in files:
        full_path = os.path.join(source_path, file)
        logger.info(f'Processing file: {full_path}')

        config = {
            'workdir': parsed_folder,
            'filename': file,
            'use_cuda': use_cuda,
            'device': device,
            'data_path': full_path,
            'ner_model_path': 'ADFLER-albert-base-v2',
            'embedding_model_path': 'swardiantara/drone-sbert',
            'birch_model_path': os.path.join(parsed_folder, 'birch_model.joblib'),
            'severity_model_path': 'swardiantara/drone-ordinal-all',
            'classifier_path': os.path.join('anomaly', 'pytorch_model.pt'),
            'utility_variable': ['severe_probs', 'sum_attributions', 'max_attributions', 'top3_attributions', 'top5_attributions', 'norm_attributions', 'snr_std', 'snr_signed', 'snr_normed', 'snr_top3', 'snr_top5', 'snr_entropy']
        }
    
        analyzer = DroneLogAnalyzer(config)
        records = analyzer.analyze()
        
        # Convert LogRecord objects to dictionaries
        serializable_records = [asdict(record) for record in records]

        # Save or process results
        output_dir = os.path.join(parsed_folder, 'record')
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{file.split('.')[0]}_records.json'), 'w') as f:
            json.dump(serializable_records, f, indent=2)


# Example usage
if __name__ == "__main__":
    main()