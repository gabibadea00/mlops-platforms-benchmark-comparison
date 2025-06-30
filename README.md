# Dataset Downloader

This script downloads datasets like Dataset-BERT, Deepcovid, AdultCensus to a specified local directory. It checks if the files already exist and saves them in organized subfolders.

## Datasets

### Dataset-BERT
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
The `-d` or `--dataset` argument chooses between 'Dataset-BERT', 'Deepcovid', 'AdultCensus'  
```
### Example usage:

```bash
# Download Deepcovid dataset
# https://github.com/shervinmin/DeepCovid/tree/masterhttps://github.com/shervinmin/DeepCovid/tree/master
python3 download_dataset.py -p ./../datasets -d deepcovid
# Download Adult Income
# https://github.com/mbeps/Adults_Income_Prediction/tree/main
python3 download_dataset.py -p ./../datasets -d adult
# Download BertSum dataset
# https://github.com/nlpyang/BertSum/tree/master
python3 download_dataset.py -p ./../datasets -d bert
```
### The resulting folder structure will look like:

```bash
datasets/
├── Deepcovid/
│   ├── test/
│   └── train/
├── AdultCensus/
│   └── adult_combined.csv
├──
```
### Requirements for this script are:  

```bash
- Python 3.7 or higher  
- `requests` Python package  
- `datasets` library from Hugging Face  

Install them using: pip install -r requirements.txt
```

### Benchmark for all these models:

```bash
    -> Rulez din mlops-platforms-benchmark-comparison/

    python3 source/benchmark_mlop.py --runs 2
```
