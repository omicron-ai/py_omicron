import os
from pathlib import Path


ROOT_DIR = Path(f"{os.path.abspath(__file__)}").parent.parent
DATA_DIR = f'{ROOT_DIR}/data'
SRC_DIR = f'{DATA_DIR}/sample_annotation.txt'
DST_DIR = f'{DATA_DIR}/sample_annotation1.json'

