import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

list_of_files = [
    "src/__init__.py",
    "src/utils.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "app.py",
    "store_index.py",
    "templates/chat.html",
    "research/medibot.ipynb"
]


for filepath in list_of_files:
    Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok = True)
        logging.info("Creating file directory {filedir} for the file {filename}")

    if (not os.path.exists(filepath) or os.path.getsize(filename)==0):
        with open(filepath, 'w' ) as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
        
    
    else:
        logging.info(f"{filename} is already available")