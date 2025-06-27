#!/usr/bin/env python3
import subprocess, os, json, time, difflib, shutil

# Structură proiect
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
            # Adaugă ulterior Metaflow/ZenML
        },
        "baseline_script": "source/general_models/AdultIncome/adult-rf.py",
        "mlops_scripts": {
            "mlflow": "source/mlops/AdultIncome/MLflow/MLflow-Adult.py",
        }
    }
}

ARTIFACT_DIRS = {
    "mlflow": "mlruns",
    "metaflow": ".metaflow",
    "zenml": ".zenml",
}

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
    total_count = 0
    total_size = 0
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            total_count += len(files)
            for f in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return {"count": total_count, "size_bytes": total_size}

def remove_artifacts(path):
    if os.path.isdir(path):
        shutil.rmtree(path)

def main():
    results = {}
    base_output = "benchmark_results"

    for model, cfg in MODELS.items():
        print(f"== Benchmark {model}")
        model_res = {}

        base_dir = os.path.join(base_output, model, "baseline")
        print("-- baseline")
        model_res["baseline"] = run_cmd(cfg["baseline_cmd"], base_dir)

        for fw, cmd in cfg["mlops"].items():
            print(f"-- {fw}")
            run_dir = os.path.join(base_output, model, fw)
            entry = run_cmd(cmd, run_dir)

            entry["code_changes"] = count_changes(cfg["baseline_script"], cfg["mlops_scripts"][fw])

            art_dir = ARTIFACT_DIRS.get(fw, "")
            entry["artifacts_before"] = count_artifacts(art_dir)

            remove_artifacts(art_dir)

            model_res[fw] = entry

        results[model] = model_res

    os.makedirs(base_output, exist_ok=True)
    with open(os.path.join(base_output, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Benchmark complet. Vezi 'benchmark_results/benchmark_results.json'")
    for fw, art in ARTIFACT_DIRS.items():
        print(f"⚠️ Artefactele pentru {fw} au fost șterse: {art}")

if __name__ == "__main__":
    main()
