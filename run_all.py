"""One command, every result -- the master script for the full experiment suite.

Runs each experiment in order with a stage-level progress bar and per-stage
timing; each script's own output (seed progress etc.) streams through unchanged.
Every stage can still be run on its own (see the README), this just chains them.

Run from the project root:  python run_all.py
"""
import subprocess, sys, time

STAGES = [
    ("run_experiments.py",   "main comparison"),
    ("run_obfuscation.py",   "obfuscation robustness"),
    ("run_sweep.py",         "sigma/k sensitivity"),
    ("run_interpolation.py", "kernel mixture"),
]


def bar(done, total, width=26):
    filled = round(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


def main():
    stages = list(STAGES)
    n = len(stages)

    t0 = time.time()
    for i, (script, label) in enumerate(stages):
        print(f"\n{bar(i, n)}  {label}  ({script})", flush=True)
        ts = time.time()
        r = subprocess.run([sys.executable, script])
        if r.returncode != 0:
            sys.exit(f"\n{script} failed with exit code {r.returncode} -- stopping here.")
        print(f"    ... {label} done in {time.time()-ts:.0f}s", flush=True)

    print(f"\n{bar(n, n)}  all done in {time.time()-t0:.0f}s "
          f"-- results/ and figures/ fully rebuilt", flush=True)


if __name__ == "__main__":
    main()