#!/usr/bin/env python3
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import argparse
import json
import pickle
import hashlib

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df_base = df.drop(['Occupation','Workclass','Marital-status','Relationship','Race','Native-country','Education','fnlwgt'], axis=1)
    occ = pd.get_dummies(df['Occupation'], prefix='Occupation')
    wc = pd.get_dummies(df['Workclass'], prefix='Workclass')
    ms = pd.get_dummies(df['Marital-status'], prefix='Marital-status')
    rel = pd.get_dummies(df['Relationship'], prefix='Relationship')
    race = pd.get_dummies(df['Race'], prefix='Race')
    nat = pd.get_dummies(df['Native-country'], prefix='Native-country')
    df = pd.concat([df_base, occ, wc, ms, rel, race, nat], axis=1)
    df['Sex'] = df['Sex'].map({'Male':1,'Female':0})
    df['Earning_potential'] = df['Earning_potential'].apply(lambda x: 1 if '>50K' in x else 0)
    return df

def split(df, seed=42):
    X = df.drop('Earning_potential', axis=1)
    y = df['Earning_potential']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_te), y_tr, y_te

def extract_main_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if tp + fn > 0 else None
    fpr = fp / (fp + tn) if fp + tn > 0 else None
    return {"accuracy": acc, "f1_score": f1, "tpr": tpr, "fpr": fpr}

def evaluate_sklearn(model, X_te, y_te):
    y_pred = model.predict(X_te)
    return extract_main_metrics(y_te, y_pred)

def hash_model(model):
    return hashlib.md5(pickle.dumps(model)).hexdigest()

def test_reproducibility(df):
    print("=== Reproducibility: Same Seed ===")
    same_seed = 42
    rf_hashes, tuned_hashes = [], []
    rf_accs, tuned_accs = [], []

    for i in range(3):
        X_tr, X_te, y_tr, y_te = split(df, seed=same_seed)

        # Baseline
        rf = RandomForestClassifier(random_state=same_seed).fit(X_tr, y_tr)
        rf_hash = hash_model(rf)
        rf_acc = rf.score(X_te, y_te)

        # Tuned
        param_grid = {'n_estimators':[50,100], 'max_depth':[5,10]}
        gs = GridSearchCV(RandomForestClassifier(random_state=same_seed), param_grid, cv=3)
        gs.fit(X_tr, y_tr)
        tuned_rf = gs.best_estimator_
        tuned_hash = hash_model(tuned_rf)
        tuned_acc = tuned_rf.score(X_te, y_te)

        rf_hashes.append((same_seed, rf_hash))
        tuned_hashes.append((same_seed, tuned_hash))
        rf_accs.append((same_seed, rf_acc))
        tuned_accs.append((same_seed, tuned_acc))

        print(f"[Initial RF]  Iter {i}, Hash: {rf_hash}, Accuracy: {rf_acc:.4f}")
        print(f"[Tuned RF]    Iter {i}, Hash: {tuned_hash}, Accuracy: {tuned_acc:.4f}")

    print("\n=== Reproducibility: Different Seeds ===")
    rf_diff_hashes, tuned_diff_hashes = [], []
    rf_diff_accs, tuned_diff_accs = [], []

    for seed in range(1, 6):
        X_tr, X_te, y_tr, y_te = split(df, seed=seed)

        rf = RandomForestClassifier(random_state=seed).fit(X_tr, y_tr)
        rf_diff_hash = hash_model(rf)
        rf_diff_acc = rf.score(X_te, y_te)

        param_grid = {'n_estimators':[50,100], 'max_depth':[5,10]}
        gs = GridSearchCV(RandomForestClassifier(random_state=seed), param_grid, cv=3)
        gs.fit(X_tr, y_tr)
        tuned_rf = gs.best_estimator_
        tuned_diff_hash = hash_model(tuned_rf)
        tuned_diff_acc = tuned_rf.score(X_te, y_te)

        rf_diff_hashes.append((seed, rf_diff_hash))
        rf_diff_accs.append((seed, rf_diff_acc))
        tuned_diff_hashes.append((seed, tuned_diff_hash))
        tuned_diff_accs.append((seed, tuned_diff_acc))

        print(f"[Initial RF]  Seed {seed}, Hash: {rf_diff_hash}, Accuracy: {rf_diff_acc:.4f}")
        print(f"[Tuned RF]    Seed {seed}, Hash: {tuned_diff_hash}, Accuracy: {tuned_diff_acc:.4f}")

    report = {
        "same_seed": {
            "initial_rf_hashes": rf_hashes,
            "initial_rf_acc": rf_accs,
            "identical_rf": all(h == rf_hashes[0] for h in rf_hashes),
            "tuned_rf_hashes": tuned_hashes,
            "tuned_rf_acc": tuned_accs,
            "identical_tuned": all(h == tuned_hashes[0] for h in tuned_hashes)
        },
        "different_seeds": {
            "initial_rf_hashes": rf_diff_hashes,
            "initial_rf_acc": rf_diff_accs,
            "tuned_rf_hashes": tuned_diff_hashes,
            "tuned_rf_acc": tuned_diff_accs,
            "all_different_rf": len(set(h for _, h in rf_diff_hashes)) == 5,
            "all_different_tuned": len(set(h for _, h in tuned_diff_hashes)) == 5
        }
    }

    with open("reproducibility_report_mlflow.json", "w") as f:
        json.dump(report, f, indent=2)

    print("📊 Raport salvat în 'reproducibility_report_mlflow.json'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_json', default="model_metrics.json")
    parser.add_argument('--test-reproducibility', action='store_true')
    args = parser.parse_args()

    df = load_and_preprocess(args.dataset_path)

    if args.test_reproducibility:
        test_reproducibility(df)
        return

    X_tr, X_te, y_tr, y_te = split(df)

    mlflow.set_experiment("AdultIncome_RF")
    results = {}

    # Baseline
    with mlflow.start_run(run_name="Baseline_RF") as run:
        baseline = RandomForestClassifier(random_state=42)
        baseline.fit(X_tr, y_tr)
        mlflow.sklearn.log_model(baseline, "model",
                                 signature=infer_signature(X_tr, baseline.predict(X_tr)))
    metrics_base = evaluate_sklearn(baseline, X_te, y_te)
    results['initial_rf'] = metrics_base

    # Tuned
    param_grid = {'n_estimators':[50,100], 'max_depth':[5,10]}
    with mlflow.start_run(run_name="Tuned_RF") as run:
        gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        mlflow.sklearn.log_model(best, "model",
                                 signature=infer_signature(X_tr, best.predict(X_tr)))
    metrics_tuned = evaluate_sklearn(best, X_te, y_te)
    results['tuned_rf'] = metrics_tuned
    results['tuned_rf']['best_params'] = gs.best_params_

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Metricile au fost salvate în '{args.output_json}'")

if __name__ == "__main__":
    main()
