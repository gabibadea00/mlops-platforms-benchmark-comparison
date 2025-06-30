import pandas as pd
import numpy as np
import json, joblib, datetime, hashlib
from typing import Tuple, Dict, List, Any
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from zenml import step, pipeline
import argparse
from sklearn.utils import shuffle
import io

def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--test-reproducibility', action='store_true')
    return vars(parser.parse_args())

@step
def load_dataset(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path)

@step
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([df.drop('Occupation', axis=1),
                    pd.get_dummies(df['Occupation']).add_prefix('Occupation_')], axis=1)
    df = pd.concat([df.drop('Workclass', axis=1),
                    pd.get_dummies(df['Workclass']).add_prefix('Workclass_')], axis=1)
    df = df.drop('Education', axis=1)
    df = pd.concat([df.drop('Marital-status', axis=1),
                    pd.get_dummies(df['Marital-status']).add_prefix('Marital-status_')], axis=1)
    df = pd.concat([df.drop('Relationship', axis=1),
                    pd.get_dummies(df['Relationship']).add_prefix('Relationship_')], axis=1)
    df = pd.concat([df.drop('Race', axis=1),
                    pd.get_dummies(df['Race']).add_prefix('Race_')], axis=1)
    df = pd.concat([df.drop('Native-country', axis=1),
                    pd.get_dummies(df['Native-country']).add_prefix('Native-country_')], axis=1)
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
    df['Earning_potential'] = df['Earning_potential'].str.contains('>50K').astype(int)
    df = df.drop('fnlwgt', axis=1)
    return df

@step
def feature_selection(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    target = "Earning_potential"
    corr = df.corr()[target].abs().sort_values()
    drop_n = int(0.8 * len(df.columns))
    to_drop = corr.iloc[:drop_n].index.tolist()
    df_sel = df.drop(columns=to_drop)
    feature_names = df_sel.drop(columns=[target]).columns.tolist()
    return df_sel, feature_names

@step
def split_data(df: pd.DataFrame, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = 'Earning_potential'
    train, test = train_test_split(df, test_size=0.2, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.drop(columns=target).values)
    X_test = scaler.transform(test.drop(columns=target).values)
    y_train = train[target].values
    y_test = test[target].values
    return X_train, X_test, y_train, y_test

def compute_metrics(model, X_te, y_te) -> Dict[str, float]:
    preds = model.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    report = classification_report(y_te, preds, output_dict=True)
    f1 = report["1"]["f1-score"]
    acc = model.score(X_te, y_te)
    return {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "tpr": float(tpr),
        "fpr": float(fpr)
    }

def hash_model(model) -> str:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return hashlib.md5(buffer.getvalue()).hexdigest()

@step
def train_rf(X_train, X_test, y_train, y_test, feature_names, label_map: Dict[int, str], seed: int) -> Tuple[Dict[str, Any], RandomForestClassifier, Dict[str, float]]:
    model = RandomForestClassifier(random_state=seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    report = classification_report(y_test, preds, output_dict=True,
                                   target_names=[label_map[0], label_map[1]])
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    metrics = compute_metrics(model, X_test, y_test)
    return ({
        "accuracy": model.score(X_test, y_test),
        "classification_report": report,
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "feature_importances": importances
    }, model, metrics)

@step
def tune_rf(X_train, X_test, y_train, y_test, feature_names, label_map: Dict[int, str], seed: int) -> Tuple[Dict[str, Any], RandomForestClassifier, Dict[str, float]]:
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 4],
        'max_features': ['sqrt', 'log2']
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=seed), param_grid, n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    preds = best.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True,
                                   target_names=[label_map[0], label_map[1]])
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    importances = dict(zip(feature_names, best.feature_importances_.tolist()))
    metrics = compute_metrics(best, X_test, y_test)
    metrics["best_params"] = grid.best_params_
    return ({
        "accuracy": best.score(X_test, y_test),
        "classification_report": report,
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "feature_importances": importances,
        "best_params": grid.best_params_
    }, best, metrics)

@step
def save_metrics(base_metrics: Dict[str, float], tuned_metrics: Dict[str, Any]):
    out = {"initial_rf": base_metrics, "tuned_rf": tuned_metrics}
    with open("model_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("✅ Saved model_metrics.json")

@step
def test_reproducibility(df: pd.DataFrame):
    X = df.drop(columns=['Earning_potential']).values
    y = df['Earning_potential'].values

    same_seed_rf_hashes, same_seed_rf_acc = [], []
    same_seed_tuned_hashes, same_seed_tuned_acc = [], []

    diff_seed_rf_hashes, diff_seed_rf_acc = [], []
    diff_seed_tuned_hashes, diff_seed_tuned_acc = [], []

    seeds = [42, 42, 42, 1, 2, 3, 4, 5]
    rf_hashes_same = []
    tuned_hashes_same = []

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        rf = RandomForestClassifier(random_state=seed).fit(X_train, y_train)
        rf_hash = hash_model(rf)
        rf_acc = rf.score(X_test, y_test)

        gs = GridSearchCV(RandomForestClassifier(random_state=seed), {
            'n_estimators': [50, 100], 'max_depth': [5, 10]
        }, cv=3)
        gs.fit(X_train, y_train)
        tuned = gs.best_estimator_
        tuned_hash = hash_model(tuned)
        tuned_acc = tuned.score(X_test, y_test)

        if seed == 42:
            same_seed_rf_hashes.append([seed, rf_hash])
            same_seed_rf_acc.append([seed, rf_acc])
            rf_hashes_same.append(rf_hash)

            same_seed_tuned_hashes.append([seed, tuned_hash])
            same_seed_tuned_acc.append([seed, tuned_acc])
            tuned_hashes_same.append(tuned_hash)
        else:
            diff_seed_rf_hashes.append([seed, rf_hash])
            diff_seed_rf_acc.append([seed, rf_acc])

            diff_seed_tuned_hashes.append([seed, tuned_hash])
            diff_seed_tuned_acc.append([seed, tuned_acc])

    report = {
        "same_seed": {
            "initial_rf_hashes": same_seed_rf_hashes,
            "initial_rf_acc": same_seed_rf_acc,
            "identical_rf": len(set(rf_hashes_same)) == 1,
            "tuned_rf_hashes": same_seed_tuned_hashes,
            "tuned_rf_acc": same_seed_tuned_acc,
            "identical_tuned": len(set(tuned_hashes_same)) == 1
        },
        "different_seeds": {
            "initial_rf_hashes": diff_seed_rf_hashes,
            "initial_rf_acc": diff_seed_rf_acc,
            "tuned_rf_hashes": diff_seed_tuned_hashes,
            "tuned_rf_acc": diff_seed_tuned_acc,
            "all_different_rf": len(set([x[1] for x in diff_seed_rf_hashes])) == len(diff_seed_rf_hashes),
            "all_different_tuned": len(set([x[1] for x in diff_seed_tuned_hashes])) == len(diff_seed_tuned_hashes)
        }
    }

    with open("reproducibility_report_zenml.json", "w") as f:
        json.dump(report, f, indent=2)

    print("📊 Saved reproducibility_report_zenml.json ✅")

@pipeline(enable_cache=False)
def main_pipeline(dataset_path: str, test_reproducibility_flag: bool):
    label_map = {0: "<=50K", 1: ">50K"}
    df = load_dataset(dataset_path)
    df2 = preprocess_data(df)
    if test_reproducibility_flag:
        test_reproducibility(df2)
    else:
        df3, feature_names = feature_selection(df2)
        X_tr, X_te, y_tr, y_te = split_data(df3, 42)
        m0, base_model, base_m = train_rf(
            X_train=X_tr,
            X_test=X_te,
            y_train=y_tr,
            y_test=y_te,
            feature_names=feature_names,
            label_map=label_map,
            seed=42
        )
        m1, final_model, tuned_m = tune_rf(
            X_train=X_tr,
            X_test=X_te,
            y_train=y_tr,
            y_test=y_te,
            feature_names=feature_names,
            label_map=label_map,
            seed=42
        )
        save_metrics(base_metrics=base_m, tuned_metrics=tuned_m)

if __name__ == "__main__":
    args = parse_args()
    main_pipeline(
        dataset_path=args["dataset_path"],
        test_reproducibility_flag=args["test_reproducibility"]
    )
