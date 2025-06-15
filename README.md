# Dataset Downloader

This script downloads datasets like SQuAD and Plantdoc to a specified local directory. It checks if the files already exist and saves them in organized subfolders.

## Datasets

### SQuAD
- TBA
  
### Plantdoc
- TBA
  
## Setup

### To set up the environment, use the following commands:

```bash
python3 -m venv venv  
source venv/bin/activate   # or venv\Scripts\activate on Windows  
pip install -r requirements.txt
```
### To download the datasets, run:

```bash
The `-p` or `--path` argument sets the output folder.  
The `-d` or `--dataset` argument chooses between `squad` or `Plantdoc`.  
The `-v` or `--version` argument is optional and only used for SQuAD (`v1.1` or `v2.0`).  

### Example usage:

```bash
# Download SQuAD v2.0
python3 download_dataset.py -p ./../datasets -d squad -v v2.0
# Download Plantdoc
python3 download_dataset.py -p ./../datasets -d plantdoc
```
### The resulting folder structure will look like:

```bash
datasets/
├── SQuAD/
│   ├── train-v2.0.json
│   └── dev-v2.0.json
├── Plantdoc/
│   ├── test
│   └── train
```
### Requirements for this script are:  

```bash
- Python 3.7 or higher  
- `requests` Python package  
- `datasets` library from Hugging Face  

Install them using: pip install -r requirements.txt
```