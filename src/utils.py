import os
import re
from pathlib import Path

def get_latest_folder(directory='.'):
    # Get all folders in the directory
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    # Filter folders matching the pattern YYYYMMDD_HHmmss
    pattern = re.compile(r'^\d{8}_\d{6}$')
    matching_folders = [f for f in folders if pattern.match(f)]
    
    # Sort by name (which effectively sorts by date and time given the format)
    matching_folders.sort()
    
    if matching_folders:
        return matching_folders[-1]
    else:
        return None