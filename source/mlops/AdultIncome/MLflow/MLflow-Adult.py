import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import argparse

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

def split(df):
    X = df.drop('Earning_potential', axis=1)
    y = df['Earning_potential']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_te), y_tr, y_te

def evaluate(model_uri, X_te, y_te, run_name):
    eval_df = pd.DataFrame(X_te, columns=[f"x{i}" for i in range(X_te.shape[1])])
    eval_df['target'] = y_te.values
    
    result = mlflow.evaluate(
        model=model_uri,
        data=eval_df,
        targets='target',
        model_type='classifier',
        evaluators=None
    )
    
    metrics = result.metrics  # dictionary of all metrics
    
    print(f"✅ Evaluation results for '{run_name}':")
    for name, val in metrics.items():
        # val may be numeric or more complex; convert to float if possible
        try:
            formatted = f"{float(val):.4f}"
        except Exception:
            formatted = str(val)
        print(f"  • {name}: {formatted}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    args = parser.parse_args()

    df = load_and_preprocess(args.dataset_path)
    X_tr, X_te, y_tr, y_te = split(df)

    mlflow.set_experiment("AdultIncome_RF")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Baseline_RF"):
        baseline = RandomForestClassifier(random_state=42)
        baseline.fit(X_tr, y_tr)
        mlflow.sklearn.log_model(baseline, "model", signature=infer_signature(X_tr, baseline.predict(X_tr)))
        run_id = mlflow.active_run().info.run_id

    evaluate(f"runs:/{run_id}/model", X_te, y_te, "Baseline_RF")

    param_grid = {'n_estimators':[50,100], 'max_depth':[5,10]}
    with mlflow.start_run(run_name="Tuned_RF"):
        gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        mlflow.sklearn.log_model(best, "model", signature=infer_signature(X_tr, best.predict(X_tr)))
        run_id2 = mlflow.active_run().info.run_id

    evaluate(f"runs:/{run_id2}/model", X_te, y_te, "Tuned_RF")

if __name__=="__main__":
    main()
