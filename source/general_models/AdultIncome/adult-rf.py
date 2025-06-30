#!/usr/bin/env python3
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import argparse
import json
import hashlib
import joblib
import os
import pickle

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([df.drop('Occupation', axis=1), pd.get_dummies(df['Occupation']).add_prefix('Occupation_')], axis=1)
    df = pd.concat([df.drop('Workclass', axis=1), pd.get_dummies(df['Workclass']).add_prefix('Workclass_')], axis=1)
    df = df.drop('Education', axis=1)
    df = pd.concat([df.drop('Marital-status', axis=1), pd.get_dummies(df['Marital-status']).add_prefix('Marital-status_')], axis=1)
    df = pd.concat([df.drop('Relationship', axis=1), pd.get_dummies(df['Relationship']).add_prefix('Relationship_')], axis=1)
    df = pd.concat([df.drop('Race', axis=1), pd.get_dummies(df['Race']).add_prefix('Race_')], axis=1)
    df = pd.concat([df.drop('Native-country', axis=1), pd.get_dummies(df['Native-country']).add_prefix('Native-country_')], axis=1)
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'Male' else 0)
    df['Earning_potential'] = df['Earning_potential'].apply(lambda x: 1 if '>50K' in x else 0)
    df = df.drop('fnlwgt', axis=1)
    return df

def feature_selection(df: pd.DataFrame, target: str = 'Earning_potential') -> pd.DataFrame:
    correlations = df.corr()[target].abs().sort_values()
    num_cols_to_drop = int(0.8 * len(df.columns))
    cols_to_drop = correlations.iloc[:num_cols_to_drop].index
    return df.drop(cols_to_drop, axis=1)

def split_data(df: pd.DataFrame, target: str, seed: int) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test

def format_metrics(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = model.score(X_test, y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    tpr = tp / (tp + fn) if tp + fn > 0 else None
    fpr = fp / (fp + tn) if fp + tn > 0 else None
    f1 = f1_score(y_test, preds, pos_label=1)
    report = classification_report(y_test, preds, output_dict=True)
    return {
        "accuracy": acc,
        "tpr": tpr,
        "fpr": fpr,
        "f1_score": f1,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "classification_report": report
    }

def train_rf(X_train, y_train, X_test, y_test, seed: int):
    model = RandomForestClassifier(random_state=seed)
    model.fit(X_train, y_train)
    return format_metrics(model, X_test, y_test), model

def tune_rf(X_train, y_train, X_test, y_test, seed: int):
    param_grid = {
        'n_estimators': [50, 100, 250],
        'max_depth': [5, 10, 30, None],
        'min_samples_split': [2, 4],
        'max_features': ['sqrt', 'log2']
    }
    gs = GridSearchCV(RandomForestClassifier(random_state=seed), param_grid, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    metrics = format_metrics(gs.best_estimator_, X_test, y_test)
    metrics["best_params"] = gs.best_params_
    return metrics, gs.best_estimator_

def hash_model(model) -> str:
    model_bytes = pickle.dumps(model)
    return hashlib.md5(model_bytes).hexdigest()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def test_reproducibility(df: pd.DataFrame):
    df = feature_selection(df, target='Earning_potential')

    print("=== Test reproducibility with same seed ===")
    same_seed = 42
    rf_hashes, rf_accs = [], []
    tuned_hashes, tuned_accs = [], []

    for i in range(3):
        X_train, X_test, y_train, y_test = split_data(df, 'Earning_potential', seed=same_seed)
        _, rf_model = train_rf(X_train, y_train, X_test, y_test, seed=same_seed)
        _, tuned_model = tune_rf(X_train, y_train, X_test, y_test, seed=same_seed)

        rf_hash = hash_model(rf_model)
        tuned_hash = hash_model(tuned_model)
        rf_acc = rf_model.score(X_test, y_test)
        tuned_acc = tuned_model.score(X_test, y_test)

        rf_hashes.append((same_seed ,rf_hash))
        rf_accs.append((same_seed, rf_acc))
        tuned_hashes.append((same_seed, tuned_hash))
        tuned_accs.append((same_seed, tuned_acc))

        print(f"[Initial RF]  Seed: {same_seed}, Hash: {rf_hash}, Accuracy: {rf_acc:.4f}")
        print(f"[Tuned RF]    Seed: {same_seed}, Hash: {tuned_hash}, Accuracy: {tuned_acc:.4f}\n")

    identical_rf = all(h == rf_hashes[0] for h in rf_hashes)
    identical_tuned = all(h == tuned_hashes[0] for h in tuned_hashes)

    print(f"→ Identical hashes for initial RF with same seed?  {identical_rf}")
    print(f"→ Identical hashes for tuned RF with same seed?    {identical_tuned}")

    print("\n=== Test difference with different seeds ===")
    rf_diff_hashes, rf_diff_accs = [], []
    tuned_diff_hashes, tuned_diff_accs = [], []

    for seed in range(1, 6):
        X_train, X_test, y_train, y_test = split_data(df, 'Earning_potential', seed=seed)
        _, rf_model = train_rf(X_train, y_train, X_test, y_test, seed=seed)
        _, tuned_model = tune_rf(X_train, y_train, X_test, y_test, seed=seed)

        rf_hash = hash_model(rf_model)
        tuned_hash = hash_model(tuned_model)
        rf_acc = rf_model.score(X_test, y_test)
        tuned_acc = tuned_model.score(X_test, y_test)

        rf_diff_hashes.append((seed,rf_hash))
        rf_diff_accs.append((seed, rf_acc))
        tuned_diff_hashes.append((seed, tuned_hash))
        tuned_diff_accs.append((seed, tuned_acc))

        print(f"[Initial RF]  Seed: {seed}, Hash: {rf_hash}, Accuracy: {rf_acc:.4f}")
        print(f"[Tuned RF]    Seed: {seed}, Hash: {tuned_hash}, Accuracy: {tuned_acc:.4f}\n")

    all_diff_rf = len(set(rf_diff_hashes)) == 5
    all_diff_tuned = len(set(tuned_diff_hashes)) == 5

    print(f"→ All initial RF hashes different?  {all_diff_rf}")
    print(f"→ All tuned RF hashes different?    {all_diff_tuned}")

    report = {
        "same_seed": {
            "initial_rf_hashes": rf_hashes,
            "initial_rf_acc": rf_accs,
            "identical_rf": identical_rf,
            "tuned_rf_hashes": tuned_hashes,
            "tuned_rf_acc": tuned_accs,
            "identical_tuned": identical_tuned,
        },
        "different_seeds": {
            "initial_rf_hashes": rf_diff_hashes,
            "initial_rf_acc": rf_diff_accs,
            "tuned_rf_hashes": tuned_diff_hashes,
            "tuned_rf_acc": tuned_diff_accs,
            "all_different_rf": all_diff_rf,
            "all_different_tuned": all_diff_tuned
        }
    }

    with open("reproducibility_report_general.json", "w") as f:
        json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_json', default="model_metrics.json")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-reproducibility', action='store_true')
    args = parser.parse_args()

    df = load_dataset(args.dataset_path)
    df = preprocess_data(df)

    if args.test_reproducibility:
        test_reproducibility(df)
        return

    df = feature_selection(df, target='Earning_potential')

    feature_names = df.drop(columns='Earning_potential').columns.tolist()
    label_map = {0: "<=50K", 1: ">50K"}

    X_train, X_test, y_train, y_test = split_data(df, 'Earning_potential', seed=args.seed)

    initial_metrics, _ = train_rf(X_train, y_train, X_test, y_test, seed=args.seed)
    tuned_metrics, _ = tune_rf(X_train, y_train, X_test, y_test, seed=args.seed)

    result = {
        "dataset": args.dataset_path,
        "feature_names": feature_names,
        "label_map": label_map,
        "initial_rf": initial_metrics,
        "tuned_rf": tuned_metrics,
    }

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2, cls=NpEncoder)

    print(f"✅ Metrice salvate în '{args.output_json}'")

if __name__ == "__main__":
    main()
