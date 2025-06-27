import pandas as pd
import numpy as np
import json, joblib, datetime
from typing import Tuple, Dict, List, Any
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from zenml import step, pipeline

LABEL_MAP = {0: "<=50K", 1: ">50K"}

@step
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

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

def compute_metrics(model, X_te, y_te) -> Dict[str, float]:
    preds = model.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # F1 pentru clasa pozitivă (1)
    report = classification_report(y_te, preds, output_dict=True)
    f1 = report["1"]["f1-score"]
    acc = model.score(X_te, y_te)
    # Return dict explicit
    return {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "tpr": float(tpr),
        "fpr": float(fpr)
    }

@step
def train_rf(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    feature_names: List[str], label_map: Dict[int, str]
) -> Tuple[Dict, RandomForestClassifier, Dict]:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True,
                                   target_names=[label_map[0], label_map[1]])
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    
    metrics = compute_metrics(model, X_test, y_test)
    
    return ({
        "accuracy": model.score(X_test, y_test),
        "classification_report": report,
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "feature_importances": importances
    }, model, metrics)

@step
def tune_rf(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    feature_names: List[str], label_map: Dict[int, str]
) -> Tuple[Dict, RandomForestClassifier, Dict]:
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
def save_metrics(
        base_metrics: Dict[str, float],
        tuned_metrics: Dict[str, Any]
    ) -> None:
    out = {"initial_rf": base_metrics, "tuned_rf": tuned_metrics}
    with open("model_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("✅ Saved model_metrics.json")

@step
def save_artifacts(
        metrics: Dict[str, Any],
        model: RandomForestClassifier,
        model_type: str
    ) -> None:

    try:
        with open("model_history.json", "r", encoding="utf-8") as f:
            history = json.load(f)
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

    fn = "final_rf_model.joblib" if model_type == "tuned" else "base_rf_model.joblib"
    joblib.dump(model, fn)

@pipeline(enable_cache=False)
def main_pipeline(
    dataset_path: str = "../../../../datasets/AdultIncome/adult_combined.csv"
    ):
    df = load_dataset(path=dataset_path)
    df2 = preprocess_data(df)
    df3, feature_names = feature_selection(df2)
    X_tr, X_te, y_tr, y_te = split_data(df3)
    m0, base_model, base_m = train_rf(
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        feature_names=feature_names,
        label_map=LABEL_MAP
    )
    # save_artifacts(metrics=m0, model=base_model, model_type="base")
    m1, final_model, tuned_m = tune_rf(
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        feature_names=feature_names,
        label_map=LABEL_MAP
    )
    # save_artifacts(metrics=m1, model=final_model, model_type="tuned")
    save_metrics(base_metrics=base_m, tuned_metrics=tuned_m)
    
if __name__ == "__main__":
    # Run from ZenML folder
    # main_pipeline()
    
    # Run from mlops-platforms-benchmark-comparison folder - for benchmark
    main_pipeline(dataset_path="datasets/AdultIncome/adult_combined.csv")

