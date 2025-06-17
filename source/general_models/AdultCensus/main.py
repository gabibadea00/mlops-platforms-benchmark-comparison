import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Union, Tuple
from pandas import DataFrame
from numpy import ndarray

import json
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_dataset(path: str) -> DataFrame:
    return pd.read_csv(path)

def preprocess_data(df: DataFrame) -> DataFrame:
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

def feature_selection(df: DataFrame, target: str = 'Earning_potential') -> DataFrame:
    correlations = df.corr()[target].abs().sort_values()
    num_cols_to_drop = int(0.8 * len(df.columns))
    cols_to_drop = correlations.iloc[:num_cols_to_drop].index
    return df.drop(cols_to_drop, axis=1)

def split_data(df: DataFrame, target: str) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    train_df, test_df = train_test_split(df, test_size=0.2)
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test
    
def train_rf(X_train, y_train, X_test, y_test, feature_names, label_map) -> Dict:
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return format_metrics(model, X_test, y_test, feature_names, label_map)

    
def tune_rf(X_train, y_train, X_test, y_test, feature_names, label_map) -> Dict:
    param_grid = {
        'n_estimators': [50, 100, 250],
        'max_depth': [5, 10, 30, None],
        'min_samples_split': [2, 4],
        'max_features': ['sqrt', 'log2']
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    metrics = format_metrics(grid_search.best_estimator_, X_test, y_test, feature_names, label_map)
    metrics["best_params"] = grid_search.best_params_
    return metrics, grid_search.best_estimator_


def format_metrics(model, X_test, y_test, feature_names: List[str], label_map: Dict[int, str]) -> Dict:
    preds = model.predict(X_test)
    report_raw = classification_report(
        y_test, preds, output_dict=True,
        target_names=[label_map[0], label_map[1]]
    )
    accuracy = model.score(X_test, y_test)

    # Matrice brută: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    # Importanță trăsături
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))

    return {
        "accuracy": accuracy,
        "classification_report": report_raw,
        "confusion_matrix": {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn)
        },
        "feature_importances": importances
    }

def main():
    df = load_dataset("./../../../datasets/AdultCensus/adult_combined.csv")
    df = preprocess_data(df)
    df = feature_selection(df, target='Earning_potential')
    
    target = 'Earning_potential'
    feature_names = df.drop(columns=target).columns.tolist()
    label_map = {0: "<=50K", 1: ">50K"}
    
    X_train, X_test, y_train, y_test = split_data(df, target)

    metrics = {}
    metrics["initial_model"] = train_rf(X_train, y_train, X_test, y_test, feature_names, label_map)
    metrics["tuned_model"], final_model = tune_rf(X_train, y_train, X_test, y_test, feature_names, label_map)

    # 📄 Salvăm metricele în JSON
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 💾 Salvăm modelul
    joblib.dump(final_model, "final_rf_model.joblib")

    # 🧾 Salvăm specificațiile de input features
    model_spec = {
        "input_feature_names": feature_names,
        "input_dimensions": len(feature_names),
        "label_mapping": label_map,
        "expected_input_shape": [None, len(feature_names)],
        "model_file": "final_rf_model.joblib"
    }

    with open("model_spec.json", "w") as f:
        json.dump(model_spec, f, indent=4)

    print("✅ Metricele, modelul și specificațiile au fost salvate în folderul curent.")

if __name__ == "__main__":
    main()
