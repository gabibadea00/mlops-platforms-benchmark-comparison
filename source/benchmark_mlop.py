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
    # "zenml": ".zenml",
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

def collect_and_move_artifacts(src, dst):
    if os.path.isdir(src):
        shutil.move(src, os.path.join(dst, os.path.basename(src)))
    else:
        for fname in ("metrics.json", "model_metrics.json", "metrics.json"):
            if os.path.exists(fname):
                os.makedirs(dst, exist_ok=True)
                shutil.move(fname, os.path.join(dst, fname))

def count_artifacts(path):
    count = size = 0
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            count += len(files)
            for f in files:
                try: size += os.path.getsize(os.path.join(root, f))
                except: pass
    return {"count": count, "size_bytes": size}

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
            src_art = ARTIFACT_DIRS.get(fw, "")
            entry["artifacts_before"] = count_artifacts(src_art)

            collect_and_move_artifacts(src_art, run_dir)

            moved_path = os.path.join(run_dir, os.path.basename(src_art))
            entry["artifacts_after"] = count_artifacts(moved_path)

            # cleanup moved artifacts under project root
            remove_artifacts(src_art)
            model_res[fw] = entry

        results[model] = model_res

    os.makedirs(base_output, exist_ok=True)
    with open(os.path.join(base_output, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Benchmark complet - rezultate în folderul 'benchmark_results/'")

if __name__ == "__main__":
    main()
