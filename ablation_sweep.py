# ablation_sweep.py — Run all 7 variants, export results CSV
"""
Variants:
  1. voice-only       (zero activity + mobility)
  2. activity-only    (zero voice + mobility)
  3. mobility-only    (zero voice + activity)  — will be ~random since mobility=zeros
  4. voice+activity   (zero mobility)
  5. voice+mobility   (zero activity)
  6. activity+mobility (zero voice)
  7. full-fusion      (all three)

Tony Stark note: "You take out one system at a time.
That's how you find out which one's load-bearing."
"""

import os
import subprocess
import csv
import re
import sys

VARIANTS = [
    ("activity-only",     ["--no_voice", "--no_mobility"]),
    ("voice-only",        ["--no_activity", "--no_mobility"]),
    ("voice+activity",    ["--no_mobility"]),
    ("activity+mobility", ["--no_voice"]),
    ("voice+mobility",    ["--no_activity"]),
    ("full-fusion",       []),
]

PYTHON = sys.executable
RESULTS_CSV = "runs/ablation_results.csv"


def run_variant(name, flags):
    """Run train.py with given ablation flags, return best val F1."""
    # Clean checkpoint dir for fresh run
    ckpt_dir = f"runs/ablation_{name}/"
    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = [
        PYTHON, "train.py",
        "--run_name", f"ablation-{name}",
        *flags,
    ]
    print(f"\n{'='*60}")
    print(f"  ABLATION: {name}")
    print(f"  Flags: {flags}")
    print(f"{'='*60}\n")

    # Set ckpt dir via env or just parse output
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    output = result.stdout + result.stderr
    print(output[-500:] if len(output) > 500 else output)

    # Parse best F1 from output
    best_f1 = 0.0
    for line in output.split("\n"):
        match = re.search(r"Best val F1:\s*([\d.]+)", line)
        if match:
            best_f1 = float(match.group(1))
        match2 = re.search(r"val_f1=([\d.]+)", line)
        if match2:
            f1 = float(match2.group(1))
            best_f1 = max(best_f1, f1)

    print(f"  → Best F1: {best_f1:.4f}")
    return best_f1


def main():
    os.makedirs("runs", exist_ok=True)
    results = []

    for name, flags in VARIANTS:
        f1 = run_variant(name, flags)
        results.append({"variant": name, "best_val_f1": f1})

    # Save CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "best_val_f1"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"  ABLATION COMPLETE")
    print(f"{'='*60}")
    for r in results:
        bar = "█" * int(r["best_val_f1"] * 40)
        print(f"  {r['variant']:<20s} F1={r['best_val_f1']:.4f}  {bar}")
    print(f"\n  Results saved → {RESULTS_CSV}")


if __name__ == "__main__":
    main()
