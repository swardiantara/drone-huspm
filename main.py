import copy
import os
import joblib
import json
import logging
from collections import defaultdict, Counter
from src.utils import get_latest_folder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SequentialPatternMiningWithDuplicates:
    """
    Implementation of a sequential pattern mining algorithm that supports
    duplicate events within itemsets (elements).
    """
    
    def __init__(self, min_support=0.5, max_pattern_length=None):
        """
        Initialize the sequential pattern miner.
        
        Args:
            min_support (float): Minimum support threshold (0.0-1.0)
            max_pattern_length (int): Maximum length of patterns to mine
        """
        self.min_support = min_support
        self.max_pattern_length = max_pattern_length
        self.frequent_patterns = []
        self.sequence_count = 0
        
    def fit(self, sequences):
        """
        Mine sequential patterns from the given sequences.
        
        Args:
            sequences (list): List of sequences, where each sequence is a list of itemsets,
                             and each itemset is a list of (event_id, utility_score) tuples
                             that may contain duplicates.
        
        Returns:
            list: List of frequent patterns with their support and utility scores
        """
        self.sequence_count = len(sequences)
        min_support_count = max(1, int(self.min_support * self.sequence_count))
        
        # Find frequent 1-patterns (single events)
        frequent_events = self._find_frequent_events(sequences, min_support_count)
        
        # Use PrefixSpan-like approach to mine patterns
        self._mine_patterns([], frequent_events, sequences, min_support_count)
        
        # Sort patterns by support and utility
        self.frequent_patterns.sort(key=lambda x: (x['support'], x['utility']), reverse=True)
        
        return self.frequent_patterns
    
    def _find_frequent_events(self, sequences, min_support_count):
        """Find all frequent single events in the sequences."""
        event_counter = Counter()
        
        for sequence in sequences:
            # Get unique events in this sequence
            sequence_events = set()
            for itemset in sequence:
                for event, _ in itemset:
                    sequence_events.add(event)
                    
            # Count each unique event only once per sequence
            for event in sequence_events:
                event_counter[event] += 1
        
        # Return events that meet minimum support
        return {event: count for event, count in event_counter.items() 
                if count >= min_support_count}
    
    def _mine_patterns(self, prefix, frequent_events, projected_db, min_support_count, level=1):
        """
        Recursively mine patterns using a PrefixSpan-like approach.
        
        Args:
            prefix (list): Current prefix pattern (list of itemsets)
            frequent_events (dict): Frequent events in the projected database
            projected_db (list): Projected database for the current prefix
            min_support_count (int): Minimum support count
            level (int): Current recursion level
        """
        # Check if we've reached the maximum pattern length
        if self.max_pattern_length and level > self.max_pattern_length:
            return
        
        # For each frequent event, extend the prefix
        for event, support_count in frequent_events.items():
            # Calculate utility score
            utility = self._calculate_utility(projected_db, event)
            
            # Create new pattern by extending prefix with this event
            # Case 1: Add as a new itemset
            s_extension = copy.deepcopy(prefix)
            s_extension.append([(event, 0)])  # Placeholder utility, will be updated
            
            # Record this pattern
            pattern_info = {
                'pattern': s_extension,
                'support': support_count / self.sequence_count,
                'support_count': support_count,
                'utility': utility
            }
            self.frequent_patterns.append(pattern_info)
            
            # Create a new projected database for this extension
            s_projected_db = self._create_projected_database(projected_db, event, 's')
            
            # Case 2: Add to the last itemset (if prefix is not empty)
            if prefix:
                i_extension = copy.deepcopy(prefix)
                i_extension[-1].append((event, 0))  # Placeholder utility
                
                # Calculate support for i-extension
                i_support = self._calculate_i_extension_support(projected_db, event, prefix)
                
                if i_support >= min_support_count:
                    # Record this pattern
                    pattern_info = {
                        'pattern': i_extension,
                        'support': i_support / self.sequence_count,
                        'support_count': i_support,
                        'utility': utility
                    }
                    self.frequent_patterns.append(pattern_info)
                    
                    # Create projected database for i-extension
                    i_projected_db = self._create_projected_database(projected_db, event, 'i', prefix)
                    
                    # Find frequent events in i-extension projected database
                    i_frequent_events = self._find_frequent_events_in_projection(
                        i_projected_db, min_support_count)
                    
                    # Recursively mine with i-extension
                    if i_frequent_events:
                        self._mine_patterns(
                            i_extension, i_frequent_events, i_projected_db, 
                            min_support_count, level + 1)
            
            # Find frequent events in s-extension projected database
            s_frequent_events = self._find_frequent_events_in_projection(
                s_projected_db, min_support_count)
            
            # Recursively mine with s-extension
            if s_frequent_events:
                self._mine_patterns(
                    s_extension, s_frequent_events, s_projected_db, 
                    min_support_count, level + 1)
    
    def _calculate_utility(self, projected_db, event):
        """
        Calculate utility score for an event across the projected database.
        In this implementation, we take the average utility score.
        """
        total_utility = 0
        count = 0
        
        for sequence in projected_db:
            for itemset in sequence:
                for e, utility in itemset:
                    if e == event:
                        total_utility += utility
                        count += 1
        
        return total_utility / max(1, count)
    
    def _calculate_i_extension_support(self, projected_db, event, prefix):
        """Calculate support for an i-extension (adding to last itemset)."""
        support_count = 0
        
        for sequence in projected_db:
            # Check if the last prefix itemset and the event appear in the same itemset
            for i, itemset in enumerate(sequence):
                if self._contains_prefix_last_itemset(itemset, prefix[-1]):
                    # Check if the event is in the same itemset
                    if any(e == event for e, _ in itemset):
                        support_count += 1
                        break
        
        return support_count
    
    def _contains_prefix_last_itemset(self, itemset, prefix_last_itemset):
        """Check if an itemset contains all events from the last itemset of prefix."""
        itemset_events = [e for e, _ in itemset]
        prefix_events = [e for e, _ in prefix_last_itemset]
        
        # Check if all prefix events appear in the itemset
        # This handles duplicates by treating each occurrence separately
        temp_itemset = itemset_events.copy()
        for event in prefix_events:
            if event in temp_itemset:
                temp_itemset.remove(event)
            else:
                return False
        return True
    
    def _create_projected_database(self, sequences, event, extension_type, prefix=None):
        """
        Create a projected database for a specific extension.
        
        Args:
            sequences (list): Current projected database
            event (str): Event to project on
            extension_type (str): 's' for sequence extension, 'i' for itemset extension
            prefix (list): Current prefix pattern (needed for i-extension)
            
        Returns:
            list: New projected database
        """
        projected_db = []
        
        for sequence in sequences:
            if extension_type == 's':
                # Find the first occurrence of the event in any itemset
                for i, itemset in enumerate(sequence):
                    for j, (e, _) in enumerate(itemset):
                        if e == event:
                            # Project from the next itemset
                            if i + 1 < len(sequence):
                                projected_sequence = sequence[i+1:]
                                if projected_sequence:
                                    projected_db.append(projected_sequence)
                            break
                    else:
                        continue
                    break
            else:  # 'i' extension
                # Find the first occurrence of the last prefix itemset
                for i, itemset in enumerate(sequence):
                    if self._contains_prefix_last_itemset(itemset, prefix[-1]):
                        # Check if the event is in the same itemset
                        for j, (e, _) in enumerate(itemset):
                            if e == event:
                                # Projection starts from the next itemset
                                if i + 1 < len(sequence):
                                    projected_sequence = sequence[i+1:]
                                    if projected_sequence:
                                        projected_db.append(projected_sequence)
                                break
                        break
        
        return projected_db
    
    def _find_frequent_events_in_projection(self, projected_db, min_support_count):
        """Find frequent events in the projected database."""
        return self._find_frequent_events(projected_db, min_support_count)


class HighUtilitySequentialPatternMiner(SequentialPatternMiningWithDuplicates):
    """
    Extension of sequential pattern mining to focus on high utility patterns.
    """
    
    def __init__(self, min_support=0.5, min_utility=0.5, max_pattern_length=None):
        """
        Initialize the high utility sequential pattern miner.
        
        Args:
            min_support (float): Minimum support threshold (0.0-1.0)
            min_utility (float): Minimum utility threshold
            max_pattern_length (int): Maximum length of patterns to mine
        """
        super().__init__(min_support, max_pattern_length)
        self.min_utility = min_utility
    
    def fit(self, sequences):
        """
        Mine high utility sequential patterns from the given sequences.
        
        Args:
            sequences (list): List of sequences, where each sequence is a list of itemsets,
                             and each itemset is a list of (event_id, utility_score) tuples
                             that may contain duplicates.
        
        Returns:
            list: List of frequent high-utility patterns with their support and utility scores
        """
        result = super().fit(sequences)
        
        # Filter patterns by utility
        high_utility_patterns = [
            pattern for pattern in result
            if pattern['utility'] >= self.min_utility
        ]
        
        # Sort by utility and then support
        high_utility_patterns.sort(key=lambda x: (x['utility'], x['support']), reverse=True)
        
        return high_utility_patterns


def preprocess_flight_logs(logs):
    """
    Convert flight logs to the format needed for sequential pattern mining.
    
    Args:
        logs (list): List of flight logs, where each log is a list of records,
                    and each record contains a list of (event_id, utility_score) tuples.
    
    Returns:
        list: List of sequences in the format needed for pattern mining
    """
    sequences = []
    
    for log in logs:
        sequence = []
        for record in log:
            # Each record becomes an itemset
            # Note: We're preserving duplicates within the itemset
            sequence.append(record)
        sequences.append(sequence)
    
    return sequences


def root_cause_analysis(patterns, anomaly_logs):
    """
    Perform root cause analysis using discovered sequential patterns.
    
    Args:
        patterns (list): High utility sequential patterns
        anomaly_logs (list): Logs with known anomalies
        
    Returns:
        list: Patterns that are likely causes of anomalies, ranked by relevance
    """
    # Find patterns that frequently precede anomalies
    causal_patterns = []
    
    for pattern in patterns:
        pattern_sequence = pattern['pattern']
        
        # Count how many anomaly logs contain this pattern
        match_count = 0
        for log in anomaly_logs:
            if contains_pattern(log, pattern_sequence):
                match_count += 1
        
        # Calculate how likely this pattern is associated with anomalies
        anomaly_association = match_count / len(anomaly_logs) if anomaly_logs else 0
        
        # Combine pattern support, utility, and anomaly association
        relevance_score = (
            pattern['support'] * 0.3 + 
            pattern['utility'] * 0.4 + 
            anomaly_association * 0.3
        )
        
        causal_patterns.append({
            'pattern': pattern_sequence,
            'support': pattern['support'],
            'utility': pattern['utility'],
            'anomaly_association': anomaly_association,
            'relevance_score': relevance_score
        })
    
    # Sort by relevance score
    causal_patterns.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return causal_patterns


def contains_pattern(sequence, pattern):
    """
    Check if a sequence contains a pattern (while handling duplicates properly).
    
    Args:
        sequence (list): A sequence (list of itemsets)
        pattern (list): A pattern (list of itemsets)
        
    Returns:
        bool: True if the sequence contains the pattern, False otherwise
    """
    if not pattern:
        return True
    
    if not sequence:
        return False
    
    # Try to find the first itemset of the pattern
    for i, itemset in enumerate(sequence):
        if contains_itemset(itemset, pattern[0]):
            # If this is the last itemset in the pattern, we're done
            if len(pattern) == 1:
                return True
            
            # Otherwise, look for the rest of the pattern in the remainder of the sequence
            if contains_pattern(sequence[i+1:], pattern[1:]):
                return True
    
    return False


def contains_itemset(itemset, pattern_itemset):
    """
    Check if an itemset contains a pattern itemset (handling duplicates).
    
    Args:
        itemset (list): An itemset (list of (event_id, utility) tuples)
        pattern_itemset (list): A pattern itemset
        
    Returns:
        bool: True if the itemset contains the pattern itemset, False otherwise
    """
    # Extract just the event IDs from both itemsets
    itemset_events = [event for event, _ in itemset]
    pattern_events = [event for event, _ in pattern_itemset]
    
    # Check if all pattern events appear in the itemset with the right frequencies
    itemset_counter = Counter(itemset_events)
    pattern_counter = Counter(pattern_events)
    
    for event, count in pattern_counter.items():
        if itemset_counter.get(event, 0) < count:
            return False
    
    return True


# Example usage
if __name__ == "__main__":
    # Example data: List of flight logs
    # Each flight log is a list of records (messages)
    # Each record is a list of (event_id, utility_score) tuples
    # example_logs = [
    #     # Flight log 1
    #     [
    #         [("E1", 0.8), ("E2", 0.4)],         # Message with 2 sentences
    #         [("E3", 0.9)],                      # Message with 1 sentence
    #         [("E1", 0.7), ("E4", 0.6), ("E1", 0.5)]  # Message with 3 sentences, E1 appears twice
    #     ],
    #     # Flight log 2
    #     [
    #         [("E1", 0.9)],
    #         [("E2", 0.5), ("E3", 0.7)],
    #         [("E4", 0.8)]
    #     ],
    #     # Flight log 3
    #     [
    #         [("E2", 0.6), ("E3", 0.8)],
    #         [("E1", 0.7), ("E1", 0.6)],         # E1 appears twice
    #         [("E4", 0.9)]
    #     ]
    # ]

    # Load pre-build sequence-db
    parsed_folder = get_latest_folder('outputs')
    utility = 'max_attributions'
    source_path = os.path.join(parsed_folder, 'sequence', utility)
    sequence_files = os.listdir(source_path)
    example_logs = []
    for sequence_file in sequence_files:
        example_log = joblib.load(os.path.join(source_path, sequence_file))
        print(example_log)
        example_logs.append(example_log['sequence'])

    # Preprocess logs
    sequences = preprocess_flight_logs(example_logs)
    
    # Mine high utility sequential patterns
    miner = HighUtilitySequentialPatternMiner(
        min_support=0.6,  # At least 30% of logs contain the pattern
        min_utility=0.9,  # Patterns with at least 0.6 average utility
        max_pattern_length=5
    )
    
    patterns = miner.fit(sequences)
    output_dir = os.path.join(parsed_folder, 'pattern')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'pattern.json'), 'w') as f:
        json.dump(patterns, f, indent=2)
    # Print discovered patterns
    print("Discovered High Utility Sequential Patterns:")
    for i, pattern in enumerate(patterns):
        # Format the pattern for readability
        pattern_str = " -> ".join(
            "[" + ", ".join(event for event, _ in itemset) + "]"
            for itemset in pattern['pattern']
        )
        print(f"{i+1}. {pattern_str}")
        print(f"   Support: {pattern['support']:.2f}, Utility: {pattern['utility']:.2f}")
    
    # Perform root cause analysis (assuming all logs are anomaly logs in this example)
    causes = root_cause_analysis(patterns, sequences)
    with open(os.path.join(output_dir, 'causes.json'), 'w') as f:
        json.dump(causes, f, indent=2)
    # Print potential root causes
    print("\nPotential Root Causes:")
    for i, cause in enumerate(causes[:3]):  # Top 3 causes
        # Format the pattern for readability
        pattern_str = " -> ".join(
            "[" + ", ".join(event for event, _ in itemset) + "]"
            for itemset in cause['pattern']
        )
        print(f"{i+1}. {pattern_str}")
        print(f"   Relevance Score: {cause['relevance_score']:.2f}")
        print(f"   Anomaly Association: {cause['anomaly_association']:.2f}")