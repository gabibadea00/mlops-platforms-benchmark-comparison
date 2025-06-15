# Dataset Downloader

This script downloads datasets like SQuAD and Tiny-CropNet to a specified local directory. It checks if the files already exist and saves them in organized subfolders.

## Setup

### To set up the environment, use the following commands:

```bash
python3 -m venv venv  
source venv/bin/activate   # or venv\Scripts\activate on Windows  
pip install -r requirements.txt
```
### To download the datasets, run:

```bash
python3 download_dataset.py -p ./../datasets -d squad -v v2.0  
python3 download_dataset.py -p ./../datasets -d cropnet --region 01003 --year 2021 --max 300

The `-p` or `--path` argument sets the output folder.  
The `-d` or `--dataset` argument chooses between `squad` or `cropnet`.  
The `-v` or `--version` argument is optional and only used for SQuAD (`v1.1` or `v2.0`).  

For CropNet, additional arguments are required:  

`--region` specifies the USDA region code (e.g., "01003"),  
`--year` specifies the crop year (e.g., 2021),  
`--max` is optional and limits the number of downloaded rows (e.g., 300).
```
### Example usage:

```bash
python3 download_dataset.py -p ./../datasets -d squad -v v2.0  
python3 download_dataset.py -p ./../datasets -d cropnet --region 01003 --year 2021 --max 300
```
### The resulting folder structure will look like:

```bash
datasets/  
├── SQuAD/  
│   ├── train-v2.0.json  
│   └── dev-v2.0.json  
└── CropNet/  
    └── tiny_cropnet_01003_2021.json
```
### Requirements for this script are:  

```bash
- Python 3.7 or higher  
- `requests` Python package  
- `datasets` library from Hugging Face  

Install them using: pip install -r requirements.txt
```