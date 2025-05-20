import os
import json
from src.data_loader import LogRecord
from typing import List, Dict, Tuple
from simpletransformers.ner import NERModel

class MessageSegmenter:
    def __init__(self, ner_model_path: str):
        self.ner_model = self._load_ner_model(ner_model_path)
        
    def _load_ner_model(self, path, **args):
        # TODO: Implement based on your NER model type
        config_file = open(os.path.join(path, 'config.json'))
        model_config = json.load(config_file)
        labels = [label for id, label in model_config['id2label'].items()]
        
        segmenter = NERModel(
                        model_config['model_type'],
                        path,
                        labels=labels,
                        use_cuda=args.use_cuda
                    )
        return segmenter
    
    def extract_spans(prediction, text=None, repair_invalid=True):
        """
        Extract valid and invalid spans from NER predictions following BIOES tagging scheme.
        
        Args:
            prediction (list): List of dictionaries, each containing a token and its BIOES tag
            text (str, optional): Original text input (not used in this implementation but included for future extensions)
            repair_invalid (bool): If True, attempt to repair invalid spans instead of just flagging them
        
        Returns:
            dict: Dictionary with three keys:
                - 'valid_spans': List of valid entity spans
                - 'repaired_spans': List of spans that were invalid but repaired (only if repair_invalid=True)
                - 'invalid_spans': List of invalid spans that couldn't be repaired
        """
        valid_spans = []
        repaired_spans = []
        invalid_spans = []
        
        # Track span information
        current_span = []
        current_entity_types = set()  # Using a set to track all entity types in a span
        current_dominant_type = None
        bioes_valid = True  # Flag to track if current span follows BIOES rules
        
        def finalize_span(span_tokens, entity_types, dominant_type, is_bioes_valid, force_invalid=False):
            """Helper function to finalize spans and categorize them appropriately"""
            if not span_tokens:
                return
            
            # Determine if span is valid, repairable, or invalid
            if not force_invalid and is_bioes_valid and len(entity_types) == 1:
                # Completely valid span
                valid_spans.append((span_tokens, list(entity_types)[0]))
            elif repair_invalid and is_bioes_valid and len(entity_types) > 1:
                # Repairable: BIOES valid but mixed entity types - use dominant or "Unknown"
                entity = dominant_type if dominant_type else "Unknown"
                repaired_spans.append((span_tokens, entity, "mixed_entity_types", list(entity_types)))
            elif repair_invalid and not is_bioes_valid and len(entity_types) == 1:
                # Repairable: Single entity type but BIOES invalid
                repaired_spans.append((span_tokens, list(entity_types)[0], "bioes_violation", None))
            else:
                # Cannot repair: Either mixed types and BIOES invalid, or repair_invalid=False
                invalid_spans.append((span_tokens, list(entity_types), "multiple_issues" if not is_bioes_valid and len(entity_types) > 1 
                                    else "bioes_violation" if not is_bioes_valid 
                                    else "mixed_entity_types"))
        
        # Process tokens and tags
        prev_tag = None
        prev_prefix = None
        expected_next = None  # What we expect next based on BIOES rules
        
        for token_dict in prediction:
            token = list(token_dict.keys())[0]
            tag = token_dict[token]
            
            # Parse tag components
            if '-' in tag and not tag == 'O':
                prefix = tag[0]  # B, I, O, E, S
                entity_type = tag[2:]  # Entity type after the prefix
            else:
                prefix = tag
                entity_type = None
            
            # Check if current token continues or breaks the current span
            new_span_needed = False
            
            # Determine if this tag violates BIOES rules for the current span
            if current_span:
                # Expected transitions based on BIOES
                if prev_prefix == 'B':
                    expected_next = ['I', 'E']
                elif prev_prefix == 'I':
                    expected_next = ['I', 'E']
                elif prev_prefix == 'E' or prev_prefix == 'S' or prev_prefix == 'O':
                    expected_next = ['B', 'S', 'O']
                
                # Check if current prefix violates the expected transition
                if prefix not in expected_next:
                    bioes_valid = False
                
                # Determine if we need to start a new span
                if prefix in ['B', 'S', 'O'] or prev_prefix in ['E', 'S', 'O']:
                    new_span_needed = True
            
            # Handle span transition if needed
            if new_span_needed and current_span:
                # Finalize the current span before starting a new one
                finalize_span(current_span, current_entity_types, current_dominant_type, bioes_valid)
                current_span = []
                current_entity_types = set()
                current_dominant_type = None
                bioes_valid = True
            
            # Process the current token based on its tag
            if prefix in ['B', 'I', 'E', 'S']:
                # Add token to current span
                current_span.append(token)
                current_entity_types.add(entity_type)
                
                # Update dominant entity type (most frequent in the span)
                # This is a simple implementation; in a real scenario, you might want to keep counts
                if not current_dominant_type:
                    current_dominant_type = entity_type
            
            # Special handling for end of span tags
            if prefix in ['E', 'S']:
                # Finalize span immediately after an ending tag
                finalize_span(current_span, current_entity_types, current_dominant_type, bioes_valid)
                current_span = []
                current_entity_types = set()
                current_dominant_type = None
                bioes_valid = True
            
            # Update tracking variables for next iteration
            prev_tag = tag
            prev_prefix = prefix
        
        # Handle any remaining span at the end
        if current_span:
            # If we have an unclosed span, it violates BIOES (should end with E or S)
            finalize_span(current_span, current_entity_types, current_dominant_type, False)
        
        # Format the results
        formatted_valid_spans = []
        for span_tokens, entity_type in valid_spans:
            formatted_valid_spans.append({
                'text': ' '.join(span_tokens),
                'tokens': span_tokens,
                'entity_type': entity_type
            })
        
        formatted_repaired_spans = []
        for span_data in repaired_spans:
            if len(span_data) == 4:  # Mixed entity types
                span_tokens, entity_type, issue_type, original_types = span_data
                formatted_repaired_spans.append({
                    'text': ' '.join(span_tokens),
                    'tokens': span_tokens,
                    'entity_type': entity_type,
                    'issue_type': issue_type,
                    'original_entity_types': original_types
                })
            else:  # BIOES violation
                span_tokens, entity_type, issue_type, _ = span_data
                formatted_repaired_spans.append({
                    'text': ' '.join(span_tokens),
                    'tokens': span_tokens,
                    'entity_type': entity_type,
                    'issue_type': issue_type
                })
        
        formatted_invalid_spans = []
        for span_data in invalid_spans:
            span_tokens, entity_types, issue_type = span_data
            formatted_invalid_spans.append({
                'text': ' '.join(span_tokens),
                'tokens': span_tokens,
                'entity_types': list(entity_types),
                'issue_type': issue_type
            })
        
        result = {
            'valid_spans': formatted_valid_spans,
            'invalid_spans': formatted_invalid_spans
        }
        
        if repair_invalid:
            result['repaired_spans'] = formatted_repaired_spans
        
        return result
    
    def segment_and_classify(self, records: List[LogRecord], verbose=False) -> List[LogRecord]:
        """Apply NER to segment messages and classify as event/non-event"""
        for record in records:
            # Process each record's raw message
            predictions, _ = self.ner_model.predict([record.raw_message])
            # Extract sentences and their event/non-event labels
            # TODO: Implement based on your NER model's output format
            result = self.extract_spans(predictions[0])
            
            if result['valid_spans']:
                for span in result['valid_spans']:
                    record.sentences.append(span['text'])
                    record.sentence_types.append(span['entity_type'])
            if result['repaired_spans']:
                for span in result['repaired_spans']:
                    record.sentences.append(span['text'])
                    record.sentence_types.append(span['entity_type'])
            if verbose:
                if result['invalid_spans']:
                    for span in result['invalid_spans']:
                        print(f"- {span['text']} ({span['entity_type']})")
        return records
    
    def _get_sentence_type(self, sentence) -> str:
        """Determine if sentence is event or non-event based on NER tags"""
        # TODO: Implement based on your tagging scheme
        pass
    
    def _get_token_labels(self, sentence) -> List[Tuple[str, str]]:
        """Extract token-level labels (B-Event, I-Event, etc.)"""
        # TODO: Implement based on your tagging scheme
        pass