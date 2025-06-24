#!/usr/bin/env python3
import subprocess, os, json, time, difflib, shutil

# Configurare
BASE_DIR = os.getcwd()
frameworks = {
    "mlflow": "rf_mlflow.py",
    "metaflow": "rf_metaflow.py",
    "zenml": "rf_zenml.py"
}
baseline = "rf_base.py"

def count_code_changes(base, mod):
    a = open(base).read().splitlines()
    b = open(mod).read().splitlines()
    diff = difflib.unified_diff(a, b, lineterm='')
    adds = dels = 0
    for line in diff:
        if line.startswith('+') and not line.startswith('+++'):
            adds += 1
        elif line.startswith('-') and not line.startswith('---'):
            dels += 1
    return {"added": adds, "deleted": dels, "total_base": len(a)}

def measure_runtime(script_path):
    start = time.time()
    proc = subprocess.run(["python3", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    duration = time.time() - start
    success = proc.returncode == 0
    return {"duration": duration, "success": success, "stdout": proc.stdout.decode(), "stderr": proc.stderr.decode()}

def count_artifacts(path):
    if not os.path.exists(path):
        return {"count": 0, "size": 0}
    count = sum(len(files) for _, _, files in os.walk(path))
    size = subprocess.check_output(["du", "-sb", path]).split()[0].decode()
    return {"count": count, "size_bytes": int(size)}

def main():
    results = {}
    # Baseline runtime
    results["baseline"] = measure_runtime(baseline)
    # Framework runs
    for fw, script in frameworks.items():
        crt = {}
        changes = count_code_changes(baseline, script)
        crt["code_changes"] = changes
        rt = measure_runtime(script)
        crt["runtime"] = rt
        # artifact tracking
        if fw == "mlflow":
            crt["artifacts"] = count_artifacts("mlruns")
        elif fw == "metaflow":
            crt["artifacts"] = count_artifacts("metaflow_data")
        elif fw == "zenml":
            crt["artifacts"] = count_artifacts("zenml_artifacts")
        results[fw] = crt

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Benchmark complet. Rezultate în benchmark_results.json")

if __name__ == "__main__":
    main()
