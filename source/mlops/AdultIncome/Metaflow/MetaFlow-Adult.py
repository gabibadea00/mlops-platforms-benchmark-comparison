#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from metaflow import FlowSpec, step, Parameter
import json
import hashlib
import pickle

class MyRFFlow(FlowSpec):
    dataset_path = Parameter(
        'dataset-path',
        help='Calea către fișierul adult_combined.csv',
        default='../../../../datasets/AdultIncome/adult_combined.csv'
    )

    test_reproducibility = Parameter(
        'test-reproducibility',
        help='Rulează testul de reproductibilitate',
        default=False,
        type=bool
    )

    @step
    def start(self):
        df = pd.read_csv(self.dataset_path)

        df = pd.concat([df.drop('Occupation', axis=1),
                        pd.get_dummies(df['Occupation'], prefix='Occupation')], axis=1)
        df = pd.concat([df.drop('Workclass', axis=1),
                        pd.get_dummies(df['Workclass'], prefix='Workclass')], axis=1)
        df = df.drop(['Education', 'fnlwgt'], axis=1)
        df = pd.concat([df.drop('Marital-status', axis=1),
                        pd.get_dummies(df['Marital-status'], prefix='Marital-status')], axis=1)
        df = pd.concat([df.drop('Relationship', axis=1),
                        pd.get_dummies(df['Relationship'], prefix='Relationship')], axis=1)
        df = pd.concat([df.drop('Race', axis=1),
                        pd.get_dummies(df['Race'], prefix='Race')], axis=1)
        df = pd.concat([df.drop('Native-country', axis=1),
                        pd.get_dummies(df['Native-country'], prefix='Native-country')], axis=1)
        df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
        df['Earning_potential'] = df['Earning_potential'].apply(lambda x: 1 if '>50K' in x else 0)

        self.df = df
        self.test_flag = self.test_reproducibility
        self.next(self.preprocess)

    @step
    def preprocess(self):
        self.X = self.df.drop('Earning_potential', axis=1).values
        self.y = self.df['Earning_potential'].values
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
        self.next(self.tune)

    @step
    def tune(self):
        param_grid = {
            'n_estimators': [50, 100, 250],
            'max_depth': [5, 10, 30, None],
            'min_samples_split': [2, 4],
            'max_features': ['sqrt', 'log2']
        }
        gs = GridSearchCV(RandomForestClassifier(random_state=42),
                          param_grid, cv=3, n_jobs=-1)
        gs.fit(self.X_train, self.y_train)
        self.best_clf = gs.best_estimator_
        self.best_params = gs.best_params_
        self.tuned_acc = self.best_clf.score(self.X_test, self.y_test)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        preds = self.best_clf.predict(self.X_test)
        acc = self.best_clf.score(self.X_test, self.y_test)
        tn, fp, fn, tp = confusion_matrix(self.y_test, preds).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else None
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        f1 = f1_score(self.y_test, preds)

        base_preds = self.clf.predict(self.X_test)
        base_acc = self.clf.score(self.X_test, self.y_test)
        base_tn, base_fp, base_fn, base_tp = confusion_matrix(self.y_test, base_preds).ravel()
        base_tpr = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else None
        base_fpr = base_fp / (base_fp + base_tn) if (base_fp + base_tn) > 0 else None
        base_f1 = f1_score(self.y_test, base_preds)

        result = {
            "initial_rf": {
                "accuracy": base_acc,
                "f1_score": base_f1,
                "tpr": base_tpr,
                "fpr": base_fpr
            },
            "tuned_rf": {
                "accuracy": acc,
                "f1_score": f1,
                "tpr": tpr,
                "fpr": fpr,
                "best_params": self.best_params
            }
        }

        with open("model_metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        print("📊 model_metrics.json salvat.")
        self.next(self.test)

    @step
    def test(self):
        def hash_model(model):
            return hashlib.md5(pickle.dumps(model)).hexdigest()
        
        if self.test_flag:
            rf_hashes, tuned_hashes, rf_accs, tuned_accs = [], [], [], []
            print("=== Reproducibility: Same Seed ===")
            for i in range(3):
                X = self.df.drop('Earning_potential', axis=1).values
                y = self.df['Earning_potential'].values
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler().fit(X_tr)
                X_tr = scaler.transform(X_tr)
                X_te = scaler.transform(X_te)

                same_seed = 42
                rf = RandomForestClassifier(random_state=same_seed).fit(X_tr, y_tr)
                rf_hashes.append((same_seed, hash_model(rf)))
                rf_accs.append((same_seed, rf.score(X_te, y_te)))

                gs = GridSearchCV(RandomForestClassifier(random_state=same_seed), {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10]
                }, cv=3)
                gs.fit(X_tr, y_tr)
                tuned_rf = gs.best_estimator_
                tuned_hashes.append((same_seed ,hash_model(tuned_rf)))
                tuned_accs.append((same_seed, tuned_rf.score(X_te, y_te)))

            print("\n=== Reproducibility: Different Seeds ===")
            rf_diff_hashes, tuned_diff_hashes, rf_diff_accs, tuned_diff_accs = [], [], [], []
            for seed in range(1, 6):
                X = self.df.drop('Earning_potential', axis=1).values
                y = self.df['Earning_potential'].values
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
                scaler = StandardScaler().fit(X_tr)
                X_tr = scaler.transform(X_tr)
                X_te = scaler.transform(X_te)

                rf = RandomForestClassifier(random_state=seed).fit(X_tr, y_tr)
                rf_diff_hashes.append((seed, hash_model(rf)))
                rf_diff_accs.append((seed, rf.score(X_te, y_te)))

                gs = GridSearchCV(RandomForestClassifier(random_state=seed), {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10]
                }, cv=3)
                gs.fit(X_tr, y_tr)
                tuned_rf = gs.best_estimator_
                tuned_diff_hashes.append((seed, hash_model(tuned_rf)))
                tuned_diff_accs.append((seed, tuned_rf.score(X_te, y_te)))

            report = {
                "same_seed": {
                    "initial_rf_acc": rf_accs,
                    "initial_rf_hashes": rf_hashes,
                    "identical_initial": len(set(rf_hashes)) == 1,
                    "tuned_rf_acc": tuned_accs,
                    "tuned_rf_hashes": tuned_hashes,
                    "identical_tuned": len(set(tuned_hashes)) == 1
                },
                "different_seeds": {
                    "initial_rf_acc": rf_diff_accs,
                    "initial_rf_hashes": rf_diff_hashes,
                    "unique_initial": len(set(rf_diff_hashes)) == 5,
                    "tuned_rf_acc": tuned_diff_accs,
                    "tuned_rf_hashes": tuned_diff_hashes,
                    "unique_tuned": len(set(tuned_diff_hashes)) == 5
                }
            }

            with open("reproducibility_report_metaflow.json", "w") as f:
                json.dump(report, f, indent=2)

            print("📊 Raport salvat în reproducibility_report_metaflow.json")
        self.next(self.end)

    @step
    def end(self):
        print("✅ Flow finalizat cu succes.")

if __name__ == '__main__':
    MyRFFlow()
