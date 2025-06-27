#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from metaflow import FlowSpec, step, Parameter, card, current
from metaflow.cards import Markdown, Table, Artifact
import json

class MyRFFlow(FlowSpec):
    dataset_path = Parameter(
        'dataset-path',
        help='Calea către fișierul adult_combined.csv',
        default='../../../../datasets/AdultIncome/adult_combined.csv'
    )

    @step
    def start(self):
        print("📁 Dataset path:", self.dataset_path)
        df = pd.read_csv(self.dataset_path)

        # transformări
        df = pd.concat([df.drop('Occupation', axis=1),
                        pd.get_dummies(df['Occupation']).add_prefix('Occupation_')], axis=1)
        df = pd.concat([df.drop('Workclass', axis=1),
                        pd.get_dummies(df['Workclass']).add_prefix('Workclass_')], axis=1)
        df = df.drop(['Education', 'fnlwgt'], axis=1)
        df = pd.concat([df.drop('Marital-status', axis=1),
                        pd.get_dummies(df['Marital-status']).add_prefix('Marital-status_')], axis=1)
        df = pd.concat([df.drop('Relationship', axis=1),
                        pd.get_dummies(df['Relationship']).add_prefix('Relationship_')], axis=1)
        df = pd.concat([df.drop('Race', axis=1),
                        pd.get_dummies(df['Race']).add_prefix('Race_')], axis=1)
        df = pd.concat([df.drop('Native-country', axis=1),
                        pd.get_dummies(df['Native-country']).add_prefix('Native-country_')], axis=1)
        df['Sex'] = df['Sex'].map({'Male':1,'Female':0})
        df['Earning_potential'] = df['Earning_potential'].apply(lambda x: 1 if '>50K' in x else 0)

        # split & scale
        self.X = df.drop('Earning_potential', axis=1).values
        self.y = df['Earning_potential'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.next(self.train)

    @step
    def train(self):
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.clf.fit(self.X_train, self.y_train)
        self.baseline_acc = self.clf.score(self.X_test, self.y_test)
        print("📌 Baseline Accuracy:", self.baseline_acc)
        self.next(self.tune)

    @step
    def tune(self):
        param_grid = {
            'n_estimators': [50,100,250],
            'max_depth': [5,10,30,None],
            'min_samples_split': [2,4],
            'max_features': ['sqrt','log2']
        }
        gs = GridSearchCV(RandomForestClassifier(random_state=42),
                          param_grid, cv=3, n_jobs=-1)
        gs.fit(self.X_train, self.y_train)
        self.best_clf = gs.best_estimator_
        self.best_params = gs.best_params_
        self.tuned_acc = self.best_clf.score(self.X_test, self.y_test)
        print("🚀 Tuned Accuracy:", self.tuned_acc)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        preds = self.best_clf.predict(self.X_test)
        self.confusion_matrix = confusion_matrix(self.y_test, preds).tolist()
        self.classif_report = classification_report(self.y_test, preds, output_dict=True)
        print("✅ Confusion Matrix:", self.confusion_matrix)
        self.next(self.end)

    @card(type='blank', id='metrics_card')
    @step
    def end(self):
        c = current.card['metrics_card']
        c.append(Markdown(f"## 🔍 Tuned Accuracy: **{self.tuned_acc:.4f}**"))
        c.append(Table([...], headers=[...]))
        c.append(Artifact(self.best_params, name="Best hyperparams"))

        # Calcul metrici baseline
        preds_b = self.clf.predict(self.X_test)
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(self.y_test, preds_b).ravel()
        recall_b = tp_b/(tp_b+fn_b)
        fpr_b = fp_b/(fp_b+tn_b)
        f1_b = self.classif_report['1']['f1-score']

        preds_t = self.best_clf.predict(self.X_test)
        cm = confusion_matrix(self.y_test, preds_t)
        tn, fp, fn, tp = cm.ravel()

        metrics_out = {
            "initial_rf": {
                "accuracy": float(self.baseline_acc),
                "f1_score": float(f1_b),
                "tpr": float(recall_b),
                "fpr": float(fpr_b),
                "confusion_matrix": {"TP": int(tp_b), "TN": int(tn_b), "FP": int(fp_b), "FN": int(fn_b)},
            },
            "tuned_rf": {
                "accuracy": float(self.tuned_acc),
                "f1_score": float(self.classif_report['1']['f1-score']),
                "tpr": float(self.classif_report['1']['recall']),
                "fpr": float(fp / (fp + tn)),
                "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
            }
        }
        with open("model_metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)
        print("✅ Metrice salvate în model_metrics.json")

if __name__ == "__main__":
    MyRFFlow()
