# Dataset Downloader

This script downloads datasets like SQuAD, Plantdoc, Deepcovid, AdultCensus to a specified local directory. It checks if the files already exist and saves them in organized subfolders.

## Datasets

### SQuAD
- TBA
  
### Plantdoc
- TBA

### Deepcovid
- TBA

### AdultCensus
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
The `-d` or `--dataset` argument chooses between 'SQuAD', 'Plantdoc', 'Deepcovid', 'AdultCensus'  
The `-v` or `--version` argument is optional and only used for SQuAD (`v1.1` or `v2.0`).  

### Example usage:

```bash
# Download SQuAD v2.0
python3 download_dataset.py -p ./../datasets -d squad -v v2.0
# Download Plantdoc
python3 download_dataset.py -p ./../datasets -d plantdoc
# Download Deepcovid dataset
# https://github.com/shervinmin/DeepCovid/tree/masterhttps://github.com/shervinmin/DeepCovid/tree/master
python3 download_dataset.py -p ./../datasets -d deepcovid
# Download Adult Income
# https://github.com/mbeps/Adults_Income_Prediction/tree/main
python3 download_dataset.py -p ./../datasets -d adult
```
### The resulting folder structure will look like:

```bash
datasets/
├── SQuAD/
│   ├── train-v2.0.json
│   └── dev-v2.0.json
├── Plantdoc/
│   ├── test/
│   └── train/
├── Deepcovid/
│   ├── test/
│   └── train/
├── AdultCensus/
│   └── adult_combined.csv
```
### Requirements for this script are:  

```bash
- Python 3.7 or higher  
- `requests` Python package  
- `datasets` library from Hugging Face  

Install them using: pip install -r requirements.txt
```