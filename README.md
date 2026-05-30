# PAPER -> https://ieeexplore.ieee.org/abstract/document/11208376

This paper presents a comparison between three popular open-source MLOps frameworks: MLflow, Metaflow, and ZenML, studied in three real-world machine learning scenarios: extractive text summarization using a BERT-based model, image analysis using Res Net, and tabular data classification using Random Forest. The comparison was carried out by developing MLOps-enhanced versions of the baseline code using each studied framework, for each of the three models. Of the three frameworks studied MLflow is notable for its low level of integration: less than 1.2% additional runtime and less than 104 lines of additional code. Although ZenML requires about 208 additional lines and increases execution time by about 19.6%, traceability is significantly improved in exchange. Furthermore, Metaflow provides strong automatic artifact versioning, which adds approximately 195 additional lines of code and increases runtime by about 110.7%. Despite these variations, reproducibility was confirmed by the fact that all platforms maintained consistent model performance under the same conditions, within a margin of 0.1 % (Table IV). Disk usage increased by about 220.4M× for MLflow, 220× for ZenML and 143.4Mx for Metaflow, these findings indicate that Metaflow provides thorough provenance at the cost of additional code and runtime overhead, ZenML strikes a reasonable balance between control and usability and MLflow is best suited for fast, low-overhead experiment tracking.

# Dataset Downloader

This script downloads datasets like Dataset-BERT, Deepcovid, AdultCensus to a specified local directory. It checks if the files already exist and saves them in organized subfolders.

## Datasets

### Dataset-BERT
### Deepcovid
### AdultCensus
  
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

-> Run from mlops-platforms-benchmark-comparison/

```bash
    python3 source/benchmark_mlop.py --runs 2
```

# Run scripts

## MLflow - Adult Income dataset

```bash
    python3 MLflow-Adult.py --dataset_path ./../../../../datasets/AdultIncome/adult_combined.csv --test-reproducibility

    mlflow ui --host 127.0.0.1 --port 5001
```
## MLflow - Deep Covid dataset

```bash
    python3 ResNet18_train.py --dataset_path ./../../../../datasets/DeepCovid/data_upload_v3 --batch_size 20 --epoch 1 --num_workers 16 --learning_rate 0.001

    mlflow ui --host 127.0.0.1 --port 5002
```

## MetaFlow - Adult Income dataset

```bash
    python3 MetaFlow-Adult.py run --with card --dataset-path ./../../../../datasets/AdultIncome/adult_combined.csv --test-reproducibility True
    # http://localhost:8324 
    python3 MetaFlow-Adult.py card server
    # python3 MetaFlow-Adult.py card view start
```

## MetaFlow - Deep Covid dataset

```bash
    python3 ResNet18_train.py --environment=conda run --with card --dataset-path ./../../../../datasets/DeepCovid/data_upload_v3 --batch_size 20 --epoch 1 --learning_rate 0.001
    # python3 ResNet18_train.py run --with card
    # http://localhost:8324 
    python3 ResNet18_train.py --environment=conda card server
    # python3 ResNet18_train.py --environment=conda card view start
```

## ZenML - Adult Income dataset

```bash
    zenml init
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

    # python3 ZenML-Adult.py
    python3 ZenML-Adult.py --dataset-path ./../../../../datasets/AdultIncome/adult_combined.csv --test-reproducibility

    zenml login --local
    zenml logout --local

        zenml server up –> pornește serverul (daemon sau Docker).
        zenml server connect <nume_server> –> conectează clientul local la server.
        zenml server logs <nume_server> –> vizualizează log-urile serverului.
        zenml server down –> oprește serverul.
```

## ZenML - Deep Covid dataset

```bash
    zenml init
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

    python3 ResNet18_train.py --dataset_path ./../../../../datasets/DeepCovid/data_upload_v3 --batch_size 20 --epoch 1 --num_workers 16 --learning_rate 0.001
    
    zenml login --local
    zenml logout --local

        zenml server up –> pornește serverul (daemon sau Docker).
        zenml server connect <nume_server> –> conectează clientul local la server.
        zenml server logs <nume_server> –> vizualizează log-urile serverului.
        zenml server down –> oprește serverul.
```
