from metaflow import FlowSpec, step, card, current, Parameter
from metaflow.cards import Markdown, VegaChart, Image, Artifact, Table, ProgressBar
import os, copy, time, argparse, json
import torch, torch.nn as nn, torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torchvision.models import ResNet18_Weights

class CovidResNetFlow(FlowSpec):
    batch_size = Parameter('batch_size', type=int, default=20)
    epochs = Parameter('epochs', type=int, default=10)
    learning_rate = Parameter('lr', type=float, default=1e-4)
    momentum = Parameter('momentum', type=float, default=0.9)
    dataset_path = Parameter('dataset-path', default='./../../../../datasets/DeepCovid/data_upload_v3')

    @step
    def start(self):
        device_name = "cpu"
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.mps.is_available():
            device_name = "mps"
        device = torch.device(device_name)
        print(f"Using device: {device}")
        self.device = device
        transforms_ = {
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
        from torch.utils.data import DataLoader
        data = {
            phase: datasets.ImageFolder(os.path.join(self.dataset_path, phase), transforms_[phase])
            for phase in ['train','test']
        }
        self.dl = {
            phase: DataLoader(data[phase], batch_size=self.batch_size, shuffle=True, num_workers=4)
            for phase in ['train','test']
        }
        self.sizes = {phase: len(data[phase]) for phase in data}
        self.class_names = data['train'].classes
        self.next(self.train)

    @card(type='blank', id='train_card', refresh_interval=1)
    @step
    def train(self):
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        for p in model.parameters(): p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, len(self.class_names))
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.fc.parameters(), lr=self.learning_rate, momentum=self.momentum)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        history = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[]}
        best_wts, best_acc = copy.deepcopy(model.state_dict()), 0.0

        # Initialize progress bars
        bars = {
            phase: ProgressBar(max=self.sizes[phase], label=f"{phase}") 
            for phase in ['train','test']
        }
        for bar in bars.values():
            current.card['train_card'].append(bar)

        for e in range(self.epochs):
            for phase in ['train', 'test']:
                bar = bars[phase]
                model.train() if phase=='train' else model.eval()
                running_loss = running_corrects = 0

                for x, y in self.dl[phase]:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase=='train'):
                        out = model(x)
                        loss = criterion(out, y)
                        if phase=='train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                        preds = out.argmax(dim=1)
                    running_loss += loss.item() * x.size(0)
                    running_corrects += torch.sum(preds == y.data)
                    # update the bar
                    bar.update(int(running_corrects))
                    current.card['train_card'].refresh()

                epoch_loss = running_loss / self.sizes[phase]
                epoch_acc = running_corrects.float().item() / self.sizes[phase]
                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_acc'].append(epoch_acc)

                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_wts)
        self.model = model
        self.history = history
        self.stats = {'best_acc': best_acc, 'history': history}
        current.card['train_card'].append(Markdown(f"## ✅ Training complete. Best test acc: {best_acc:.4f}"))
        self.next(self.evaluate)

    @card(type='blank', id='eval_card')
    @step
    def evaluate(self):
        model, dl = self.model, self.dl['test']
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        softmax = nn.Softmax(dim=1)
        for x, y in dl:
            x = x.to(self.device)
            with torch.no_grad():
                out = model(x)
                prob = softmax(out)[:,1].cpu().numpy()
                pred = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(y.numpy())
            all_probs.extend(prob)

        cm = confusion_matrix(all_labels, all_preds)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)

        current.card['eval_card'].append(
            VegaChart({
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "data": {"values": [{"fpr": a, "tpr": b} for a, b in zip(fpr, tpr)]},
                "mark": "line",
                "encoding": {"x": {"field": "fpr"}, "y": {"field": "tpr"}}
            })
        )
        current.card['eval_card'].append(Markdown(f"**Confusion Matrix**:\n```\n{cm}\n```\n**AUC**: {roc_auc:.3f}"))

        self.confusion = cm.tolist()
        self.roc_auc = roc_auc
        self.next(self.end)

    @step
    def end(self):
        with open("metrics.json","w") as f:
            json.dump({
                'confusion_matrix': self.confusion,
                'roc_auc': self.roc_auc,
                'stats': self.stats
            }, f, indent=2)
        print("✅ Flow done.")

if __name__ == "__main__":
    CovidResNetFlow()
