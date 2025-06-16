import os
import json
import requests
import pandas as pd
from argparse import ArgumentParser
import tensorflow as tf
from datetime import datetime
import zipfile
import tarfile

def parse_example(example_proto):
    return tf.io.parse_single_example(example_proto, {
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "ndvi_mean": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "region": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "year": tf.io.FixedLenFeature([], tf.int64, default_value=0)
    })

def tfrecord_to_csv(tfrecord_dir, output_csv):
    if os.path.exists(output_csv):
        print(f"⏩ CSV already exists: {output_csv}")
        return output_csv

    print(f"🔄 Converting TFRecord files from {tfrecord_dir} to {output_csv}")
    records = []
    for filename in os.listdir(tfrecord_dir):
        if filename.endswith(".tfrecord"):
            tfrecord_path = os.path.join(tfrecord_dir, filename)
            raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
            for raw_record in raw_dataset:
                example = parse_example(raw_record)
                parsed = {k: v.numpy().decode() if isinstance(v.numpy(), bytes) else v.numpy() for k, v in example.items()}
                records.append(parsed)
    if records:
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved CSV to: {output_csv}")
        return output_csv
    else:
        print("⚠️ No records found in TFRecords.")
        return None

def filter_csv_by_date(csv_path, start_date=None, end_date=None):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        print(f"⚠️ No 'date' column in {csv_path}")
        return

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    df.to_csv(csv_path, index=False)
    print(f"✅ Date filtered: {csv_path} (rows: {len(df)})")

def download_squad(path, version="v2.0"):
    urls = {
        "v1.1": {
            "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
            "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
        },
        "v2.0": {
            "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
            "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
        }
    }
    if version not in urls:
        raise ValueError("Unsupported SQuAD version. Use 'v1.1' or 'v2.0'.")
    squad_dir = os.path.join(path, "SQuAD")
    os.makedirs(squad_dir, exist_ok=True)
    for split, url in urls[version].items():
        filename = os.path.join(squad_dir, os.path.basename(url))
        if os.path.exists(filename):
            print(f"✔ SQuAD {split} ({version}) already downloaded: {filename}")
            continue
        print(f"⬇ Downloading {split} set ({version}) to {filename}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✅ Saved {split} set to {filename}")
        else:
            raise Exception(f"❌ Failed to download {url}: Status code {response.status_code}")

def download_plantdoc(path):
    url = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
    zip_path = os.path.join(path, "plantdoc_dataset.zip")

    print("⬇ Downloading PlantDoc dataset...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Saved PlantDoc zip to: {zip_path}")
    else:
        raise Exception(f"❌ Failed to download PlantDoc: Status code {response.status_code}")

    extract_dir = os.path.join(path, "PlantDoc")
    if not os.path.exists(extract_dir):
        import zipfile
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"✅ Extracted to: {extract_dir}")
            os.remove(zip_path)
            print(f"🗑️ Deleted zip file: {zip_path}")
        except zipfile.BadZipFile:
            print("❌ Error: ZIP is invalid. Please re-download manually.")
    else:
        print("✔ PlantDoc dataset already extracted.")

def download_deepcovid(path):
    url = "https://www.dropbox.com/scl/fi/ajy4i9u4bjt4ho3dz4l37/data_upload_v3.zip?rlkey=kyh5oz91vykk7cao6jiip4dyn&dl=1"
    zip_path = os.path.join(path, "deepcovid_dataset.zip")
    print("⬇ Downloading DeepCovid dataset...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Saved DeepCovid zip to: {zip_path}")
    else:
        raise Exception(f"❌ Failed to download DeepCovid: Status code {response.status_code}")
    extract_dir = os.path.join(path, "DeepCovid")
    if not os.path.exists(extract_dir):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"✅ Extracted to: {extract_dir}")
            os.remove(zip_path)
            print(f"🗑️ Deleted zip file: {zip_path}")
        except zipfile.BadZipFile:
            print("❌ Error: ZIP is invalid. Please re-download manually.")
    else:
        print("✔ DeepCovid dataset already extracted.")
      
def download_adult(path):
    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    files = ["adult.data", "adult.test", "adult.names"]
    dest = os.path.join(path, "AdultCensus")
    os.makedirs(dest, exist_ok=True)

    for fname in files:
        url = base + fname
        out = os.path.join(dest, fname)
        if os.path.exists(out):
            print(f"✔ Already downloaded: {fname}")
            continue
        print(f"⬇ Downloading {fname}...")
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(out, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            print(f"✅ Saved {fname}")
        else:
            print(f"❌ Failed to download {fname}: {r.status_code}")

    # Optional: combină data + test într-un CSV unic
    combined = os.path.join(dest, "adult_combined.csv")
    if not os.path.exists(combined):
        df_data = pd.read_csv(os.path.join(dest, "adult.data"),
                              header=None, names=None,
                              na_values=' ?', skipinitialspace=True)
        df_test = pd.read_csv(os.path.join(dest, "adult.test"),
                              header=0, names=df_data.columns,
                              na_values=' ?', skipinitialspace=True)
        df = pd.concat([df_data, df_test], ignore_index=True)
        df.to_csv(combined, index=False)
        print(f"✅ Combined CSV: {combined}")
    else:
        print("✔ Combined CSV already exists.")
  
def main(args):
    os.makedirs(args.path, exist_ok=True)
    if args.dataset.lower() == "squad":
        download_squad(args.path, version=args.version or "v2.0")
    elif args.dataset.lower() == "plantdoc":
        download_plantdoc(args.path)
    elif args.dataset.lower() == "deepcovid":
        download_deepcovid(args.path)
    elif args.dataset.lower() == "adult":
        download_adult(args.path)
    else:
        raise ValueError("Invalid dataset name! Options: squad, plantdoc, deepcovid")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', required=True)
    parser.add_argument('-d', '--dataset', required=True,
                    choices=["squad","plantdoc","deepcovid","adult"])
    parser.add_argument('--version', help="SQuAD version")
    args = parser.parse_args()
    main(args)
