from typing import List, Dict, Tuple
from src.data_loader import LogRecord

class SequenceBuilder:
    # def __init__(self, utility: str):
    #     self.utility = utility

    def build_sequences(self, records: List[LogRecord], utility: str) -> Dict:
        """Construct the final sequential database"""
        sequence = []

        event_count = 0
        for record in records:
            element = []
            if not record.eventIds:
                continue
            
            for i in range(len(record.sentences)):
                event_count += 1
                element.append((record.eventIds[i], getattr(record, utility)[i]))
            
            sequence.append(element)
        
        return {
            'metadata': {
                'total_records': len(records),
                'total_event': event_count
            },
            'sequence': sequence
        }