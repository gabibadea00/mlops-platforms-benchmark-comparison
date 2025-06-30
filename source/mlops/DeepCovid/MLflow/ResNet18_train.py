from __future__ import print_function, division
import os, copy, time, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import json
from torchvision.models import ResNet18_Weights
import mlflow, mlflow.pytorch
from mlflow.models.signature import infer_signature
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser(description='COVID-19 Detection from X-ray Images')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='num data loader workers')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--dataset_path', type=str, default='./data/', help='data directory')
    return parser.parse_args()

def build_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),
    }

def load_data(data_dir, transforms_, batch_size, num_workers):
    datasets_ = {x: datasets.ImageFolder(os.path.join(data_dir, x), transforms_[x])
                 for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(
                        datasets_[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(datasets_[x]) for x in datasets_}
    class_names = datasets_['train'].classes
    return dataloaders, dataset_sizes, class_names

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title: plt.title(title)
    plt.pause(0.001)

def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "tpr": tpr,
        "fpr": fpr
    }

def collect_metrics(training_stats: dict, class_names: list, args, best_acc: float, device: str, y_true, y_pred):
    final_metrics = compute_metrics(y_true, y_pred)
    
    metrics = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "device": device,
        "model": "resnet18",
        "frozen_layers": "all except fc",
        "optimizer": "SGD",
        "scheduler": {
            "type": "StepLR",
            "step_size": 7,
            "gamma": 0.1
        },
        "accuracy": final_metrics["accuracy"],
        "f1_score": final_metrics["f1_score"],
        "tpr": final_metrics["tpr"],
        "fpr": final_metrics["fpr"],
        "class_names": class_names,
        "training_stats": training_stats,
        "best_test_accuracy": round(best_acc, 4)
    }

    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Metricile au fost salvate în 'model_metrics.json'")

def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, device, num_epochs=20):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    stats = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n' + '-'*10)
        for phase in ['train','test']:
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs,1)
                    loss = criterion(outputs, labels)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float().item() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == "train":
                stats["train_loss"].append(round(epoch_loss, 4))
                stats['train_acc'].append(round(epoch_acc, 4))
            else:
                stats["test_loss"].append(round(epoch_loss, 4))
                stats["test_acc"].append(round(epoch_acc, 4))

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        print()
    mlflow.log_metrics({
        f"{phase}_loss": epoch_loss,
        f"{phase}_acc": float(epoch_acc)
    }, step=epoch)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best test Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, stats, best_acc

def visualize_model(dataloaders, model, device, class_names, num_images=64):
    model.eval()
    images_shown = 0
    fig = plt.figure()
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs,1)
            for j in range(inputs.size()[0]):
                images_shown += 1
                ax = plt.subplot(num_images//8, 8, images_shown)
                ax.axis('off')
                ax.set_title(f'pred: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])
                if images_shown == num_images:
                    return

def build_model():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

def main():
    mlflow.set_experiment("COVID19_ResNet18")
    mlflow.pytorch.autolog()
    
    args = parse_args()
    transforms_ = build_transforms()
    dataloaders, dataset_sizes, class_names = load_data(args.dataset_path, transforms_, args.batch_size, args.num_workers)
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.mps.is_available():
        device_name = "mps"
    device = torch.device(device_name)
    print(f"Using device: {device}")

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=args.learning_rate, momentum=args.momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model, stats, best_acc = train_model(
        dataloaders, dataset_sizes, model,
        criterion, optimizer, scheduler, device,
        num_epochs=args.epochs
    )
    
    mlflow.pytorch.log_model(
        model,
        name="covid_resnet18",
        signature=infer_signature(
            torch.randn(1,3,224,224).numpy(),
            model(torch.randn(1,3,224,224).to(device)).cpu().detach().numpy()
        ),
        input_example=torch.randn(1,3,224,224).numpy()
    )
    
    # Evaluate on test set
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    collect_metrics(stats, class_names, args, best_acc, str(device), y_true, y_pred)
    # torch.save(model, f'covid_resnet18_epoch{args.epochs}.pt')
    # visualize_model(dataloaders, model, device, class_names)


if __name__ == '__main__':
    main()
