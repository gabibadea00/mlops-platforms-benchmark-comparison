import os
import requests
from argparse import ArgumentParser

def download_cropnet(path):
    pass

def download_squad(path, version = "v2.0"):
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

    os.makedirs(path, exist_ok=True)

    for split, url in urls[version].items():
        filename = os.path.join(path, os.path.basename(url))
        print(f"Downloading {split} set ({version}) to {filename}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Saved {split} set to {filename}")
        else:
            raise Exception(f"Failed to download {url}: Status code {response.status_code}")

def main(args):
    if not os.path.exists(args.path):
        raise ValueError("Invalid path: Path doesn't exist. Check spelling.")
    
    if not os.path.isdir(args.path):
        raise ValueError("Invalid path: Not a directory")

    dataset_name = args.dataset.lower()

    if dataset_name == "cropnet":
        download_cropnet(args.path)
    elif dataset_name == "squad":
        download_squad(args.path)
    else:
        raise ValueError("Invalid dataset name! Options: cropnet or squad")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, help="Path where the dataset should be downloaded")
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Dataset name: \"CropNet\" or \"SQuAD\"")
    parser.add_argument('-v', '--version', type=str, required=False, help="Version of the SQuAD dataset: v1.1 or v2.0")
    args = parser.parse_args()

    result = main(args)
    exit(result)