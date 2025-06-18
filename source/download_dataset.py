import os
import requests
import pandas as pd
from argparse import ArgumentParser
import tensorflow as tf
import zipfile

def parse_example(example_proto):
    return tf.io.parse_single_example(example_proto, {
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "ndvi_mean": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "region": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "year": tf.io.FixedLenFeature([], tf.int64, default_value=0)
    })

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
    files = ["adult.data", "adult.test"]
    dest = os.path.join(path, "AdultIncome")
    os.makedirs(dest, exist_ok=True)

    for fname in files:
        out = os.path.join(dest, fname)
        if os.path.exists(out):
            print(f"✔ {fname} deja descărcat.")
            continue
        print(f"⬇ Descărcare {fname}...")
        r = requests.get(base + fname, stream=True)
        if r.status_code == 200:
            with open(out, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            print(f"✅ Salvare {fname}")
        else:
            print(f"❌ Eroare {fname}: {r.status_code}")

    combined = os.path.join(dest, "adult_combined.csv")
    if os.path.exists(combined):
        print("✔ adult_combined.csv deja există.")
        return combined

    col_names = [
        "Age", "Workclass", "fnlwgt", "Education", "Education-num",
        "Marital-status", "Occupation", "Relationship", "Race",
        "Sex", "Capital-gain", "Capital-loss", "Hours-per-week",
        "Native-country", "Earning_potential"
    ]

    df_data = pd.read_csv(os.path.join(dest, "adult.data"), names=col_names,
                          sep=",", na_values=' ?', skipinitialspace=True)
    df_test = pd.read_csv(os.path.join(dest, "adult.test"), names=col_names,
                          sep=",", skiprows=1, na_values=' ?', skipinitialspace=True)

    df = pd.concat([df_data, df_test], ignore_index=True)
    df.to_csv(combined, index=False)
    print(f"✅ adult_combined.csv creat ({df.shape[0]} rânduri, {df.shape[1]} coloane)")
    return combined

def main(args):
    os.makedirs(args.path, exist_ok=True)
    if args.dataset.lower() == "deepcovid":
        download_deepcovid(args.path)
    elif args.dataset.lower() == "adult":
        download_adult(args.path)
    else:
        raise ValueError("Invalid dataset name! Options: adult, deepcovid")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', required=True)
    parser.add_argument('-d', '--dataset', required=True,
                    choices=["deepcovid","adult"])
    args = parser.parse_args()
    main(args)
