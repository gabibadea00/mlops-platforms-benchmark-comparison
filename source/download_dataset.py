import os
import requests
import pandas as pd
from argparse import ArgumentParser
from datasets import load_dataset

def download_cropnet_streamed_fallback(path, region="01003", year=2021, max_rows=500):
    cropnet_dir = os.path.join(path, "CropNet")
    os.makedirs(cropnet_dir, exist_ok=True)

    output_file = os.path.join(cropnet_dir, f"tiny_cropnet_streamed_{region}_{year}.csv")
    if os.path.exists(output_file):
        print(f"✔ Streamed subset already exists: {output_file}")
        return

    try:
        print(f"⬇ Attempting to stream Tiny-CropNet from HuggingFace (region={region}, year={year})...")
        dataset_iter = load_dataset("fudong03/Tiny-CropNet", split="train", streaming=True)

        selected_samples = []
        fallback_samples = []

        for sample in dataset_iter:
            if sample.get("region") == region and sample.get("year") == year:
                selected_samples.append(sample)
                if len(selected_samples) >= max_rows:
                    break
            elif len(fallback_samples) < max_rows:
                fallback_samples.append(sample)

        if selected_samples:
            df = pd.DataFrame(selected_samples)
            print(f"✅ Matched {len(df)} rows for region={region}, year={year}")
        else:
            print(f"⚠ No exact match for region={region}, year={year}. Falling back to first {max_rows} available rows.")
            df = pd.DataFrame(fallback_samples)

        df.to_csv(output_file, index=False)
        print(f"✅ Streamed dataset saved to: {output_file} (rows: {len(df)})")

    except Exception as e:
        print(f"❌ Failed to download streamed Tiny-CropNet: {e}")

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

def main(args):
    if not os.path.exists(args.path):
        raise ValueError("Invalid path: Path doesn't exist.")
    if not os.path.isdir(args.path):
        raise ValueError("Invalid path: Not a directory")

    dataset_name = args.dataset.lower()

    if dataset_name == "cropnet":
        download_cropnet_streamed_fallback(
            path=args.path,
            region=args.region,
            year=args.year,
            max_rows=args.max
        )
    elif dataset_name == "squad":
        version = args.version if args.version else "v2.0"
        download_squad(args.path, version=version)
    else:
        raise ValueError("Invalid dataset name! Options: cropnet or squad")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help="Path where the dataset should be downloaded")
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help="Dataset name: 'CropNet' or 'SQuAD'")
    parser.add_argument('-v', '--version', type=str, required=False,
                        help="SQuAD version: 'v1.1' or 'v2.0'")
    parser.add_argument('--region', type=str, default="01003",
                        help="Region code for Tiny-CropNet filtering")
    parser.add_argument('--year', type=int, default=2021,
                        help="Year for Tiny-CropNet filtering")
    parser.add_argument('--max', type=int, default=500,
                        help="Max number of rows to download from Tiny-CropNet")
    args = parser.parse_args()

    main(args)
