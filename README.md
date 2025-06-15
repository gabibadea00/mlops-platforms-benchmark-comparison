# Dataset Downloader

This script downloads datasets like **SQuAD** and **Tiny-CropNet** to a specified local directory.
It checks if the files already exist and saves them in organized subfolders.

## Setup

```bash

python3 -m venv venv
source venv/bin/activate  
# or venv\Scripts\activate on Windows
pip install -r requirements.txtß

python3 download_dataset.py  -p ./../datasets -d squad -v v2.0
python3 download_dataset.py -p ./../datasets -d cropnet --region 01003 --year 2021 --max 300
ß
-p or --path: output folder

-d or --dataset: squad or cropnet

-v or --version: optional, for SQuAD only (v1.1 or v2.0)

📦 Output structure:
    datasets/
    ├── SQuAD/
    │   ├── train-v2.0.json
    │   └── dev-v2.0.json
    └── CropNet/
        └── tiny_cropnet_01003_2021.json

🔒 Requirements:
    -> Python 3.7+
    -> requests
    -> datasets from Hugging Face

```
