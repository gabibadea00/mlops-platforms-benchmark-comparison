#!/usr/bin/env python3
import subprocess, os, json, time, difflib

BASE_MODELS = ["rf", "resnet", "bert"]
FRAMEWORKS = ["mlflow", "metaflow", "zenml"]
BASE_DIR = os.getcwd()

def count_changes(base, mod):
    a = open(base).read().splitlines()
    b = open(mod).read().splitlines()
    diff = difflib.unified_diff(a, b, lineterm='')
    adds = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
    dels = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
    return {"added": adds, "deleted": dels, "base_lines": len(a)}

def measure_runtime(cmd):
    start = time.time()
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return {"duration": time.time() - start, "success": proc.returncode == 0, "stderr": proc.stderr}

def count_artifacts(path):
    if not os.path.isdir(path):
        return {"count": 0, "size_bytes": 0}
    count = sum(len(files) for _, _, files in os.walk(path))
    size = int(subprocess.check_output(f"du -sb {path}", shell=True).split()[0])
    return {"count": count, "size_bytes": size}

def benchmark():
    results = {}
    for model in BASE_MODELS:
        baseline = os.path.join("general_models", model, f"{model}_base.py")
        results[model] = {"baseline": measure_runtime(f"python3 {baseline}")}
        for fw in FRAMEWORKS:
            script = os.path.join("mlops", model, f"{model}_{fw}.py")
            changes = count_changes(baseline, script)
            rt = measure_runtime(f"python3 {script}")
            art_dir = {
                "mlflow": "mlruns",
                "metaflow": "metaflow_data",
                "zenml": "zenml_artifacts"
            }[fw]
            arts = count_artifacts(art_dir)
            results[model][fw] = {"code_changes": changes, "runtime": rt, "artifacts": arts}
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Benchmark saved to benchmark_results.json")

if __name__ == "__main__":
    benchmark()
