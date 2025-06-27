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

def split_data(df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
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
    f1 = f1_score(y_test, preds, pos_label=1)  # F1 pentru clasa pozitivă
    report = classification_report(y_test, preds, output_dict=True)
    return {
        "accuracy": acc,
        "tpr": tpr,
        "fpr": fpr,
        "f1_score": f1,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "classification_report": report
    }

def train_rf(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return format_metrics(model, X_test, y_test), model

def tune_rf(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [50, 100, 250],
        'max_depth': [5, 10, 30, None],
        'min_samples_split': [2, 4],
        'max_features': ['sqrt', 'log2']
    }
    gs = GridSearchCV(RandomForestClassifier(random_state=42),
                      param_grid, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    metrics = format_metrics(gs.best_estimator_, X_test, y_test)
    metrics["best_params"] = gs.best_params_
    return metrics

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_json', default="model_metrics.json")
    args = parser.parse_args()

    df = load_dataset(args.dataset_path)
    df = preprocess_data(df)
    df = feature_selection(df, target='Earning_potential')

    feature_names = df.drop(columns='Earning_potential').columns.tolist()
    label_map = {0: "<=50K", 1: ">50K"}

    X_train, X_test, y_train, y_test = split_data(df, 'Earning_potential')

    # Train
    initial_metrics, _ = train_rf(X_train, y_train, X_test, y_test)
    # Tune
    tuned_metrics = tune_rf(X_train, y_train, X_test, y_test)

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
