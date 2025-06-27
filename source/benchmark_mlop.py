#!/usr/bin/env python3
import subprocess, os, json, time, difflib, shutil

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
                "run --pipeline main_pipeline "
            )
        },
        "baseline_script": "source/general_models/AdultIncome/adult-rf.py",
        "mlops_scripts": {
            "mlflow": "source/mlops/AdultIncome/MLflow/MLflow-Adult.py",
            "metaflow": "source/mlops/AdultIncome/Metaflow/MetaFlow-Adult.py",
            "zenml": "source/mlops/AdultIncome/ZenML/ZenML-Adult.py",
        }
    }
}

ARTIFACT_DIRS = { 
    "mlflow": "mlruns",
    "metaflow": ".metaflow",
    "zenml": ".zen",
}
METRICS_FILES = ["model_metrics.json", "metrics.json"]

TARGET_METRICS = ["accuracy", "f1_score", "tpr", "fpr"]

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
    return {"count": total_count, "size_bytes": total_size}

def remove_artifacts(path):
    if os.path.isdir(path):
        shutil.rmtree(path)

def load_metrics(run_dir):
    metrics = {}
    for fname in METRICS_FILES:
        for fp in [os.path.join(run_dir, fname), fname]:
            if os.path.exists(fp):
                try:
                    raw = json.load(open(fp))
                    for prefix in ("initial_rf", "tuned_rf"):
                        if prefix in raw:
                            block = raw[prefix]
                            metrics[prefix] = {m: block.get(m, None) for m in TARGET_METRICS}
                    os.remove(fp)
                    print(f"✅ {fp} citit și șters.")
                    return metrics
                except Exception as e:
                    print(f"❌ Eroare la citirea {fp}: {e}")
    print(f"⚠️ Nu am găsit fișier de metrici în {run_dir} sau cwd")
    return metrics

def main():
    results = {}
    base_output = "benchmark_results"
    os.makedirs(base_output, exist_ok=True)

    for model, cfg in MODELS.items():
        print(f"=== Benchmark {model}")
        model_res = {}

        # Baseline
        base_dir = os.path.join(base_output, model, "baseline")
        print("-- baseline")
        entry_base = run_cmd(cfg["baseline_cmd"], base_dir)
        entry_base["metrics"] = load_metrics(base_dir)
        model_res["baseline"] = entry_base
        # MLOps
        for fw, cmd in cfg["mlops"].items():
            print(f"-- {fw}")
            run_dir = os.path.join(base_output, model, fw)
            entry = run_cmd(cmd, run_dir)
            entry["code_changes"] = count_changes(cfg["baseline_script"], cfg["mlops_scripts"][fw])
            entry["artifacts"] = count_artifacts(ARTIFACT_DIRS.get(fw, ""))
            entry["metrics"] = load_metrics(run_dir)
            remove_artifacts(ARTIFACT_DIRS.get(fw, ""))
            model_res[fw] = entry

        results[model] = model_res

    os.makedirs(base_output, exist_ok=True)
    with open(os.path.join(base_output, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Benchmark complet – vezi 'benchmark_results/benchmark_results.json'")
    for fw, art in ARTIFACT_DIRS.items():
        print(f"⚠️ Artefactele pentru {fw} au fost șterse: {art}")

if __name__ == "__main__":
    main()
