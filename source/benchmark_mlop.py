#!/usr/bin/env python3
import subprocess, os, json, time, difflib, shutil, glob, statistics
import argparse
import os
import json
import shutil

MODELS = {
    "AdultIncome": {
        "baseline_cmd": (
            "python3 source/general_models/AdultIncome/adult-rf.py "
            "--dataset_path datasets/AdultIncome/adult_combined.csv"
        ),
        "mlops": {
            "mlflow": (
                "python3 source/mlops/AdultIncome/MLflow/MLflow-Adult.py "
                "--dataset_path datasets/AdultIncome/adult_combined.csv"
            ),
            "metaflow": (
                "python3 source/mlops/AdultIncome/Metaflow/MetaFlow-Adult.py "
                "run --with card "
                "--dataset-path datasets/AdultIncome/adult_combined.csv"
            ),
            "zenml": (
                "zenml init && "
                "python3 source/mlops/AdultIncome/ZenML/ZenML-Adult.py "
                "--dataset-path datasets/AdultIncome/adult_combined.csv"
            )
        },
        "baseline_script": "source/general_models/AdultIncome/adult-rf.py",
        "mlops_scripts": {
            "mlflow": "source/mlops/AdultIncome/MLflow/MLflow-Adult.py",
            "metaflow": "source/mlops/AdultIncome/Metaflow/MetaFlow-Adult.py",
            "zenml": "source/mlops/AdultIncome/ZenML/ZenML-Adult.py",
        }
    },
    "DeepCovid": {
        "baseline_cmd": (
            "python3 source/general_models/DeepCovid/ResNet18_train.py "
            "--dataset_path datasets/DeepCovid/data_upload_v3 "
            "--batch_size 20 "
            "--epoch 1 "
            "--num_workers 16 "
            "--learning_rate 0.001"
        ),
        "mlops": {
            "mlflow": (
                "python3 source/mlops/DeepCovid/MLflow/ResNet18_train.py "
                "--dataset_path datasets/DeepCovid/data_upload_v3 "
                "--batch_size 20 "
                "--epoch 1 "
                "--num_workers 16 "
                "--learning_rate 0.001"
            ),
            "metaflow": (
                "python3 source/mlops/DeepCovid/Metaflow/ResNet18_train.py "
                "--environment=conda "
                "run --with card "
                "--dataset-path datasets/DeepCovid/data_upload_v3 "
                "--batch_size 20 "
                "--epoch 1 "
                "--learning_rate 0.001"
            ),
            "zenml": (
                "zenml init && "
                "python3 source/mlops/DeepCovid/ZenML/ResNet18_train.py  "
                "--dataset_path datasets/DeepCovid/data_upload_v3 "
                "--batch_size 20 "
                "--epoch 1 "
                "--num_workers 16 "
                "--learning_rate 0.001 "
            )
        },
        "baseline_script": "source/general_models/DeepCovid/ResNet18_train.py",
        "mlops_scripts": {
            "mlflow": "source/mlops/DeepCovid/MLflow/ResNet18_train.py",
            "metaflow": "source/mlops/DeepCovid/Metaflow/ResNet18_train.py",
            "zenml": "source/mlops/DeepCovid/Metaflow/ResNet18_train.py",
        }
    }
}

ARTIFACT_DIRS = { 
    "mlflow": "mlruns",
    "metaflow": ".metaflow",
    "zenml": ".zen",
}

METRICS_FILES = ["model_metrics.json", "metrics.json"]

TARGET_METRICS = [
    "duration",
    "artifact_count",
    "artifact_size",
    "accuracy",
    "f1_score",
    "tpr",
    "fpr",
    "code_changes.added",
    "code_changes.deleted",
    "code_changes.base_lines"
]

def count_changes(base, mod):
    a, b = open(base).read().splitlines(), open(mod).read().splitlines()
    diff = difflib.unified_diff(a, b, lineterm='')
    adds = sum(1 for l in diff if l.startswith('+') and not l.startswith('+++'))
    dels = sum(1 for l in diff if l.startswith('-') and not l.startswith('---'))
    return {"added": adds, "deleted": dels, "base_lines": len(a)}

def run_cmd(cmd, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = round(time.time() - start, 2)
    with open(os.path.join(out_dir, "stdout.txt"), "w") as f: f.write(proc.stdout)
    with open(os.path.join(out_dir, "stderr.txt"), "w") as f: f.write(proc.stderr)
    return {"duration": duration, "success": proc.returncode == 0}

def count_artifacts(path):
    total_count = total_size = 0
    total_count = total_size = 0
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            total_count += len(files)
            for f in files:
                try: total_size += os.path.getsize(os.path.join(root, f))
                except: pass
                try: total_size += os.path.getsize(os.path.join(root, f))
                except: pass
    return {"artifact_count": total_count, "artifact_size": total_size}

def remove_artifacts(path):
    if os.path.isdir(path):
        shutil.rmtree(path)

def load_metrics(run_dir):
    if not os.path.isdir(run_dir):
        print(f"⚠️ Directorul {run_dir} nu există.")
        return {}
    for fname in METRICS_FILES:
        for fp in [os.path.join(run_dir, fname), fname]:
            if os.path.exists(fp):
                try:
                    raw = json.load(open(fp))
                    os.remove(fp)
                    print(f"✅ {fp} citit și șters.")
                    # 1. RF-style
                    if isinstance(raw, dict):
                        out = {}
                        if "initial_rf" in raw or "tuned_rf" in raw:
                            for pref in ("initial_rf", "tuned_rf"):
                                if pref in raw:
                                    block = raw[pref]
                                    out[pref] = {m: block.get(m) for m in TARGET_METRICS}
                            return out
                        # 2. top-level metrics
                        top = {m: raw.get(m) for m in TARGET_METRICS if m in raw}
                        if top:
                            return top
                        # 3. fallback: orice altă cheie numerică / dict
                        metrics = {k: v for k, v in raw.items() if isinstance(v, (int, float, str, list, dict))}
                        return metrics
                    else:
                        print(f"⚠️ Format JSON neașteptat în {fp}.")
                except Exception as e:
                    print(f"❌ Eroare la citirea {fp}: {e}")
    print(f"⚠️ Nu am găsit fișier de metrici în {run_dir}")
    return {}

def aggregate_metrics():
    files = glob.glob("benchmark_results/benchmark_results_*.json")
    agg = {}

    for fn in files:
        run = json.load(open(fn))
        for model, frameworks in run.items():
            agg.setdefault(model, {})
            for fw, data in frameworks.items():
                agg[model].setdefault(fw, {}).setdefault("metrics_list", [])
                metrics = data.get("metrics", {})
                
                # Normalize nested + top-level metrics
                def flatten(d, parent_key=''):
                    items = {}
                    for k, v in d.items():
                        new_key = f"{parent_key}.{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.update(flatten(v, new_key))
                        else:
                            items[new_key] = v
                    return items

                flat_metrics = flatten(metrics)
                agg[model][fw]["metrics_list"].append(flat_metrics)

    summary = {}
    for model, frameworks in agg.items():
        summary[model] = {}
        for fw, data in frameworks.items():
            mlist = data["metrics_list"]
            summary[model][fw] = {}

            for metric in TARGET_METRICS:
                vals = []
                for m in mlist:
                    val = m.get(metric)
                    if isinstance(val, (int, float)):
                        vals.append(val)
                if vals:
                    summary[model][fw][metric] = {
                        "min": min(vals),
                        "max": max(vals),
                        "mean": statistics.mean(vals),
                        "stddev": statistics.pstdev(vals)
                    }

    out = {
        "aggregated_over_runs": len(files),
        "frameworks": summary
    }
    with open("benchmark_results/aggregate_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("✅ Wrote aggregate_summary.json")

def remove_none_fields(d):
    if isinstance(d, dict):
        return {k: remove_none_fields(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [remove_none_fields(x) for x in d]
    else:
        return d

def benchmark_mlops(runs: int = 5):
    base_output = "benchmark_results"
    os.makedirs(base_output, exist_ok=True)

    for i in range(1, runs + 1):
        print(f"\n🔁 === RUN {i}")
        results = {}
        run_output = os.path.join(base_output, f"benchmark_results_{i}.json")

        for model, cfg in MODELS.items():
            print(f"\n=== Benchmark {model}")
            model_res = {}

            # Baseline
            base_dir = os.path.join(base_output, model, "baseline", f"run_{i}")
            print("-- baseline")
            entry_base = run_cmd(cfg["baseline_cmd"], base_dir)
            entry_base["code_changes"] = {"added": 0, "deleted": 0, "base_lines": 0}
            entry_base["artifacts"] = count_artifacts(ARTIFACT_DIRS.get("baseline", ""))  # sau ""
            entry_base["metrics"] = load_metrics(base_dir)
            entry_base["metrics"].update({
                "duration": entry_base["duration"],
                "code_changes": entry_base["code_changes"],
                "artifact_count": entry_base["artifacts"]["artifact_count"],
                "artifact_size": entry_base["artifacts"]["artifact_size"]
            })
            model_res["baseline"] = entry_base

            # MLOps Frameworks
            for fw, cmd in cfg["mlops"].items():
                print(f"-- {fw}")
                run_dir = os.path.join(base_output, model, fw, f"run_{i}")
                entry = run_cmd(cmd, run_dir)
                entry["code_changes"] = count_changes(cfg["baseline_script"], cfg["mlops_scripts"][fw])
                entry["artifacts"] = count_artifacts(ARTIFACT_DIRS.get(fw, ""))
                entry["metrics"] = load_metrics(run_dir)
                entry["metrics"].update({
                    "duration": entry["duration"],
                    "code_changes": entry["code_changes"],
                    "artifact_count": entry["artifacts"]["artifact_count"],
                    "artifact_size": entry["artifacts"]["artifact_size"]
                })
                remove_artifacts(ARTIFACT_DIRS.get(fw, ""))
                model_res[fw] = entry

            results[model] = model_res

        with open(run_output, "w") as f:
            json.dump(remove_none_fields(results), f, indent=2)
        print(f"✅ Saved results: {run_output}")

    print("\n📊 === Aggregating all results...")
    aggregate_metrics()

def rf_seed_reproducibility():
    print("\n🔁 === REPRODUCIBILITY CHECK (RandomForest, AdultIncome) ===")
    cmds = {
        "baseline": "python3 source/general_models/AdultIncome/adult-rf.py "
                    "--dataset_path datasets/AdultIncome/adult_combined.csv --test-reproducibility",
        "mlflow": "python3 source/mlops/AdultIncome/MLflow/MLflow-Adult.py "
                  "--dataset_path datasets/AdultIncome/adult_combined.csv --test-reproducibility",
        "metaflow": "python3 source/mlops/AdultIncome/Metaflow/MetaFlow-Adult.py "
                    "run --with card "
                    "--dataset-path datasets/AdultIncome/adult_combined.csv "
                    "--test-reproducibility True",
        "zenml": "python3 source/mlops/AdultIncome/ZenML/ZenML-Adult.py "
                 "--dataset-path datasets/AdultIncome/adult_combined.csv "
                 "--test-reproducibility"
    }
    base_output = "benchmark_results"
    os.makedirs(base_output, exist_ok=True)
    
    output_dir = "benchmark_results/reproducibility_outputs"
    os.makedirs(output_dir, exist_ok=True)

    for name, cmd in cmds.items():
        print(f"\n▶️ Running reproducibility for: {name.upper()}")
        out_path = os.path.join(output_dir, f"{name}_reproducibility.json")

        # Rulează comanda și capturează rezultatul
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"❌ Error in {name}: {result.stderr}")
            continue

        # Caută fișierul JSON generat automat de scriptul curent
        possible_files = [
            "reproducibility_report_general.json",
            "reproducibility_report_mlflow.json",
            "reproducibility_report_metaflow.json",
            "reproducibility_report_zenml.json"
        ]
        found = False
        for f in possible_files:
            if os.path.exists(f):
                shutil.move(f, out_path)
                print(f"✅ Saved {f} → {out_path}")
                found = True
                break
        if not found:
            print(f"⚠️ No reproducibility report found for {name}")
        
        for model, cfg in MODELS.items():
            for fw, cmd in cfg["mlops"].items():
                remove_artifacts(ARTIFACT_DIRS.get(fw, ""))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--runs", type=int, default=2, help="Number of runs")
    args = p.parse_args()
    
    # Reproducibility test for RF for MLops
    rf_seed_reproducibility()
    
    # BenchMark evaluation for all models and all mlops
    benchmark_mlops(args.runs)

