import os
import time
import copy
import json
import torch
import argparse
import torchvision
import numpy as np
from typing import Dict, Any, Tuple
import torch.nn as nn
import torch.optim as optim
from zenml import step, pipeline, log_metadata
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

@step
def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else
              "mps"    if torch.backends.mps.is_available() else "cpu")
    settings = vars(args)
    settings["device"] = device
    log_metadata(metadata={"parsed_args": settings})
    return settings

@step
def load_data(settings: Dict[str, Any]) -> Dict[str, Any]:
    dataset_path = settings["dataset_path"]
    bs = settings["batch_size"]
    tfm = {
        'train': transforms.Compose([
            transforms.Resize(224), transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    }
    datasets_ = {phase: datasets.ImageFolder(os.path.join(dataset_path, phase), tfm[phase])
                 for phase in ["train","test"]}
    loaders = {phase: DataLoader(
        datasets_[phase], batch_size=bs,
        shuffle=(phase=="train"), num_workers=settings["num_workers"])
        for phase in ["train","test"]}
    sizes = {p: len(datasets_[p]) for p in loaders}
    classes = datasets_["train"].classes
    log_metadata(metadata={"dataset_sizes": sizes, "classes": classes})
    return {"loaders": loaders, "sizes": sizes, "classes": classes}

@step
def build_model() -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in model.parameters(): p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

@step
def train_model(
    data: Dict[str, Any],
    model: nn.Module,
    settings: Dict[str, Any]
) -> Tuple[nn.Module, Dict[str, Any]]:
    device = settings["device"]
    loaders, sizes = data["loaders"], data["sizes"]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=settings["learning_rate"],
                          momentum=settings["momentum"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = model.to(device)
    stats = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    since = time.time()

    for epoch in range(settings["epochs"]):
        for phase in ["train","test"]:
            model.train() if phase=="train" else model.eval()
            running_loss = running_corrects = 0
            for x, y in loaders[phase]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=="train"):
                    out = model(x); loss = criterion(out, y)
                    if phase=="train":
                        loss.backward(); optimizer.step()
                preds = out.argmax(dim=1)
                running_loss += loss.item()*x.size(0)
                running_corrects += torch.sum(preds==y).item()
            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_corrects / sizes[phase]
            stats[f"{phase}_loss"].append(epoch_loss)
            stats[f"{phase}_acc"].append(epoch_acc)
            if phase=="test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
        scheduler.step()

    model.load_state_dict(best_wts)
    
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loaders["test"]:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    tpr = float(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    fpr = float(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    total_time = time.time() - since
    result = {
        "stats": stats,
        "best_test_acc": float(best_acc),
        "train_time_s": total_time,
        "accuracy": acc,
        "f1_score": f1,
        "tpr": tpr,
        "fpr": fpr
    }

    log_metadata(metadata=result)
    return model, result

@step
def save_outputs(model: nn.Module, train_info: Dict[str, Any]) -> None:
    # torch.save(model.state_dict(), "covid_resnet18_final.pth")
    # log_metadata(metadata={"saved_model": "covid_resnet18_final.pth"})
    with open("model_metrics.json", "w") as f:
        json.dump(train_info, f, indent=4)

# ---------- Pipeline ----------

@pipeline(enable_cache=False)
def covid_pipeline():
    settings = parse_args()
    data = load_data(settings)
    model = build_model()
    trained, info = train_model(data, model, settings)
    save_outputs(trained, info)

if __name__ == "__main__":
    covid_pipeline()
