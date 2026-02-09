# config.py - Shared paths configuration
import os

# Data directory
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# CSV file path
CSV_FILE = os.path.join(DATA_DIR, 'cstr_simulation_data.csv')

