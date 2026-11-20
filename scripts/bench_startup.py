"""Repeatable timings for the things a user waits on.

    python scripts/bench_startup.py                 # launch timings, 5 runs
    python scripts/bench_startup.py --runs 3 --train --json bench.json

startup   process start of run_local.py until GET /config answers 200
          (the healthcheck, imports, UI build and server start; no browser)
import    `import app` alone, in a fresh interpreter
train     scripts/full_smoke_test.py end to end: every modality trains, evaluates
          and exports on the bundled fixtures (about 3 minutes; --train-runs,
          default 1)

Each figure is the median of --runs fresh processes, with min and max, so a
change can be compared before and after on the same machine. The first run
after a reboot or a file copy is slower (cold disk cache); --runs >= 3 and
the median absorb that.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from http.client import HTTPConnection
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.runtime_setup import find_free_port  # noqa: E402


def _answers(port: int) -> bool:
    conn = HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        conn.request("GET", "/config")
        return conn.getresponse().status == 200
    except OSError:
        return False
    finally:
        conn.close()


def time_startup(venv_dir: str, timeout: float = 180.0) -> float:
    port = find_free_port(7900)
    cmd = [sys.executable, str(ROOT / "run_local.py"), "--venv-dir", venv_dir,
           "--server-port", str(port), "--no-browser"]
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while time.perf_counter() - start < timeout:
            if proc.poll() is not None:
                raise RuntimeError(f"run_local.py exited early with code {proc.returncode}")
            if _answers(port):
                return time.perf_counter() - start
            time.sleep(0.05)
        raise RuntimeError(f"No answer on port {port} within {timeout:.0f}s")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def time_command(cmd: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - start


def summarize(samples: list[float]) -> dict:
    return {
        "median_s": round(statistics.median(samples), 2),
        "min_s": round(min(samples), 2),
        "max_s": round(max(samples), 2),
        "runs": len(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--venv-dir", default=str(Path(sys.prefix)),
                        help="Environment run_local.py should use (default: this interpreter's).")
    parser.add_argument("--train", action="store_true", help="Also time the end-to-end training workload.")
    parser.add_argument("--train-runs", type=int, default=1)
    parser.add_argument("--json", help="Write the results to this file as well.")
    args = parser.parse_args()

    results = {
        "startup": summarize([time_startup(args.venv_dir) for _ in range(args.runs)]),
        "import": summarize([time_command([sys.executable, "-c", "import app"]) for _ in range(args.runs)]),
    }
    if args.train:
        results["train"] = summarize(
            [time_command([sys.executable, "scripts/full_smoke_test.py"]) for _ in range(args.train_runs)]
        )

    for name, stats in results.items():
        print(f"{name:8s} median {stats['median_s']:6.2f}s  (min {stats['min_s']:.2f}, max {stats['max_s']:.2f}, "
              f"{stats['runs']} runs)")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
