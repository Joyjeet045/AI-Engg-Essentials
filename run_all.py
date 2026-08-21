"""Run every project and fail if any of them does.

Each project's main() exits non-zero when its correctness gate trips, so this is
the whole test suite. If a mask, a block table, a rollback, a copy-on-write or an
all-reduce breaks, one of these returns a non-zero code and so does this.
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECTS = [
    ("01-kv-cache", "KV_Cache.py"),
    ("02-continuous-batching", "InFlight_Batching.py"),
    ("03-paged-attention", "PagedAttention.py"),
    ("04-vllm-architecture", "vLLM_Architecture.py"),
    ("05-speculative-decoding", "speculative_decoding.py"),
    ("06-quantization", "quantization.py"),
    ("07-prefix-caching", "prefix_caching.py"),
    ("08-benchmark-harness", "benchmark.py"),
    ("09-fused-paged-attention", "fused_paged_attention.py"),
    ("10-disaggregated-serving", "disaggregated.py"),
]


def main():
    root = Path(__file__).resolve().parent
    failures = []

    print(f"{'project':<28}{'result':>8}{'time':>9}")
    for folder, script in PROJECTS:
        start = time.perf_counter()
        # Run from inside the folder, which also proves each one is standalone.
        result = subprocess.run(
            [sys.executable, script],
            cwd=root / folder,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - start
        ok = result.returncode == 0
        print(f"{folder:<28}{'ok' if ok else 'FAILED':>8}{elapsed:>8.1f}s")
        if not ok:
            failures.append((folder, result.stdout, result.stderr))

    for folder, out, err in failures:
        print(f"\n----- {folder} -----")
        print(out.strip()[-2000:])
        print(err.strip()[-2000:])

    if failures:
        raise SystemExit(f"\n{len(failures)} of {len(PROJECTS)} projects failed")
    print(f"\nall {len(PROJECTS)} projects passed")


if __name__ == "__main__":
    main()
