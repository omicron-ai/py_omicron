import os
from pathlib import Path

# --- UTILITIES
_indent_num = 1
_indent_char = "\t"
INDENT = f"{_indent_char * _indent_num}"

# --- DIRECTORIES
ROOT_DIR = Path(f"{os.path.abspath(__file__)}").parent.parent
DATA_DIR = f'{ROOT_DIR}/data'
IMAGE_DIR = f'{ROOT_DIR}/images'
DIALOG_DIR = f"{DATA_DIR}/dialog"
SCRIPT_DIR = f"{DATA_DIR}/script"

# --- SAMPLE ANNOTATION FILE PATHS
SRC_PATH = f'{DATA_DIR}/sample_annotation.txt'
JSON_PATH = f'{DATA_DIR}/sample_annotation1.json'
SEM_PATH = f'{DATA_DIR}/sample_dialog_representations.txt'
COMP_PATH = f'{DATA_DIR}/sample_composite_annotation.json'

# --- SAMPLE DIALOG ANNOTATION FILE PATHS
A0 = f"{DIALOG_DIR}/sample_agent0_dialog.json"
A1 = f"{DIALOG_DIR}/sample_agent1_dialog.json"
COMPOSITE = f"{DIALOG_DIR}/sample_dialog.json"


