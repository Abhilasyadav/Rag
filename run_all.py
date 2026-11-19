import subprocess
import sys


def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable] + cmd, check=False)
    if result.returncode != 0:
        print(f"\n  [!] {label} exited with code {result.returncode}. Check output above.")
    return result.returncode == 0


if __name__ == "__main__":
    steps = [
        (["verifiers.py"],           "Task 01 — Deterministic Verifiers smoke-test"),
        (["calibrate.py"],           "Task 03 — Judge Calibration"),
        (["experiments.py", "--variant", "A"], "Task 04A — Baseline Experiment"),
        (["experiments.py", "--variant", "B"], "Task 04B — Chain-of-Thought Experiment"),
        (["experiments.py", "--variant", "C"], "Task 04C — Few-Shot Experiment"),
        (["analyze.py"],             "Task 05 — Comparative Diff Analysis"),
    ]

    all_ok = True
    for cmd, label in steps:
        ok = run(cmd, label)
        if not ok:
            all_ok = False

    print("\n" + ("All tasks completed successfully." if all_ok else
                  "Some tasks failed — review errors above."))
