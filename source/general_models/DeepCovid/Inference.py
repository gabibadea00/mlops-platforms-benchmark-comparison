#### Inference COVID-19 Detection - Modular Version

from __future__ import print_function
import os, copy, time, glob, argparse, pickle
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser(description='COVID-19 Detection Inference')
    parser.add_argument('--test_covid_path', type=str, default='./data/test/covid/')
    parser.add_argument('--test_non_covid_path', type=str, default='./data/test/non/')
    parser.add_argument('--trained_model_path', type=str, default='./covid_resnet18_epoch2.pt')
    parser.add_argument('--cut_off_threshold', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=0)
    return parser.parse_args()

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model

def get_data_paths(test_covid, test_non):
    covid_files = glob.glob(os.path.join(test_covid, '*'))
    non_files = glob.glob(os.path.join(test_non, '*'))
    return covid_files, non_files

def make_loader(imsize=224):
    return transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

def image_loader(loader, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0).to(device)
    return image

def predict(model, loader_fn, files, device):
    sm = nn.Softmax(dim=1)
    probs = []
    preds = []
    for img_path in files:
        img = image_loader(loader_fn, img_path, device)
        out = model(img)
        prob = sm(out).cpu().detach().numpy()[0,0]
        pred = out.argmax(dim=1).item()
        probs.append(prob)
        preds.append(pred)
    return np.array(probs), np.array(preds)

def find_sens_spec(covid_prob, noncovid_prob, thresh):
    sens = (covid_prob >= thresh).sum() / len(covid_prob)
    spec = (noncovid_prob < thresh).sum() / len(noncovid_prob)
    print(f"sensitivity= {sens:.3f}, specificity= {spec:.3f}")
    return sens, spec

def plot_confusion_and_roc(covid_prob, noncovid_prob, thresh):
    covid_pred = (covid_prob > thresh).astype(int)
    non_pred = (noncovid_prob > thresh).astype(int)
    y_test = np.concatenate([np.ones_like(covid_prob), np.zeros_like(noncovid_prob)])
    y_pred = np.concatenate([covid_pred, non_pred])

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=['COVID','Non'], columns=['COVID','Non'])
    sn.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.clf()

    y_scores = np.concatenate([covid_prob, noncovid_prob])
    auc = roc_auc_score(y_test, y_scores)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.plot(fpr, tpr, label=f"ResNet18 AUC= {auc:.3f}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('ROC_covid19.png')
    plt.clf()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.trained_model_path, device)
    covid_files, non_files = get_data_paths(args.test_covid_path, args.test_non_covid_path)
    loader_fn = make_loader()
    covid_prob, _ = predict(model, loader_fn, covid_files, device)
    noncovid_prob, _ = predict(model, loader_fn, non_files, device)
    _ = find_sens_spec(covid_prob, noncovid_prob, args.cut_off_threshold)
    plot_confusion_and_roc(covid_prob, noncovid_prob, args.cut_off_threshold)
    print("✅ Inference and evaluation completed.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total time:", time.time() - start_time)
