import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple, Any
import json
import joblib
import datetime

from zenml import step, pipeline

@step
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@step
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # transformări
    df = pd.concat([...], axis=1)  # cum ai avut înainte
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
    df['Earning_potential'] = df['Earning_potential'].str.contains('>50K').astype(int)
    df = df.drop('fnlwgt', axis=1)
    return df

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
def feature_selection(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    target = "Earning_potential"
    corr = df.corr()[target].abs().sort_values()
    drop_n = int(0.8 * len(df.columns))
    to_drop = corr.iloc[:drop_n].index.tolist()
    df_sel = df.drop(columns=to_drop)
    feature_names = df_sel.drop(columns=[target]).columns.tolist()
    return df_sel, feature_names

@step
def split_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = 'Earning_potential'
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.drop(columns=target).values)
    X_test = scaler.transform(test.drop(columns=target).values)
    y_train = train[target].values
    y_test = test[target].values
    return X_train, X_test, y_train, y_test

@step
def train_rf(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    feature_names: List[str], label_map: Dict[int, str]
) -> Tuple[Dict, RandomForestClassifier]:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True,
                                   target_names=[label_map[0], label_map[1]])
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    return ({
        "accuracy": model.score(X_test, y_test),
        "classification_report": report,
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "feature_importances": importances
    }, model)

@step
def tune_rf(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    feature_names: List[str], label_map: Dict[int, str]
) -> Tuple[Dict, RandomForestClassifier]:
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 4],
        'max_features': ['sqrt', 'log2']
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    preds = best.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True,
                                   target_names=[label_map[0], label_map[1]])
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    importances = dict(zip(feature_names, best.feature_importances_.tolist()))
    return ({
        "accuracy": best.score(X_test, y_test),
        "classification_report": report,
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "feature_importances": importances,
        "best_params": grid.best_params_
    }, best)


@step
def save_artifacts(metrics: Dict[str, Any], model: RandomForestClassifier, model_type: str) -> None:
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    if model_type == "tuned":
        joblib.dump(model, "final_rf_model.joblib")
    else:
        joblib.dump(model, "base_rf_model.joblib")
@step
def save_artifacts(metrics: Dict[str, Any],
                   model: RandomForestClassifier,
                   model_type: str) -> None:

    try:
        with open("model_history.json", "r", encoding="utf-8") as f:
            history = json.load(f)  # așteptăm un list de dict-uri
            if not isinstance(history, list):
                history = []
    except FileNotFoundError:
        history = []

    entry = {
        "session_start": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model_type,
        "metrics": metrics
    }
    history.append(entry)

    with open("model_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    filename = "final_rf_model.joblib" if model_type == "tuned" else "base_rf_model.joblib"
    joblib.dump(model, filename)
    
@pipeline(enable_cache=False)
def main_pipeline(path: str, label_map: Dict[int, str]):
    df = load_dataset(path)
    df2 = preprocess_data(df)
    df3, feature_names = feature_selection(df2)
    X_tr, X_te, y_tr, y_te = split_data(df3)
    m0, base_model = train_rf(
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        feature_names=feature_names,
        label_map=label_map
    )
    save_artifacts(m0, base_model, "base")
    m1, final_model = tune_rf(
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        feature_names=feature_names,
        label_map=label_map
    )
    save_artifacts(m1, final_model, "tuned")

if __name__ == "__main__":
    main_pipeline(
        path="../../../../datasets/AdultIncome/adult_combined.csv",
        label_map={0: "<=50K", 1: ">50K"}
    )

