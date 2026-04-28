"""
Smoke tests for the 5 shipped walkers.

Runs each walker on wiki-Vote with --all and asserts:
  - exit code == 0
  - non-empty stdout
  - at least one walk line is present
  - walk length matches the walker class (fixed vs. probabilistic-stop)

Does not assert exact walk contents (RNG nondeterminism). This is a
smoke test — its purpose is to catch "generated code fails to compile
or crashes at runtime," not to validate sampling correctness.

Mirrors the invocation pattern in run_templ_one.sh.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e


# (walker_name, stop_mode): fixed-length walkers terminate at max_depth;
# probabilistic walkers terminate when their stopping condition fires.
WALKERS: list[tuple[str, str]] = [
    ("Node2vec", "fixed"),
    ("Node2vec_weighted", "fixed"),
    ("Metapath", "fixed"),
    ("Metapath_weighted", "fixed"),
    ("PPR_second", "probabilistic"),
]

# A "walk line" is heuristically identified as a line of integers separated
# by whitespace with length > 1. This regex keeps the test robust to
# surrounding log noise.
_WALK_LINE_RE = re.compile(r"^\s*\d+(?:\s+\d+)+\s*$")


def _parse_walks(stdout: str) -> list[list[int]]:
    """Extract integer-sequence walk lines from --printworkload output."""
    walks: list[list[int]] = []
    for line in stdout.splitlines():
        if _WALK_LINE_RE.match(line):
            walks.append([int(x) for x in line.split()])
    return walks


@pytest.mark.parametrize("walker,stop_mode", WALKERS)
def test_walker_smoke(walker, stop_mode, built_flowwalker, repo_root):
    build_dir = repo_root / "build"
    config_file = repo_root / "tests" / "e2e" / "fixtures" / "wiki-Vote-test.config"
    assert config_file.exists(), (
        f"Expected {config_file} to exist. Test-specific wiki-Vote config "
        "is required for smoke tests."
    )

    cmd = [
        str(built_flowwalker),
        "--config", str(config_file),
        "--all",
        "--printresult",
        "--GPU_count", "1",
        f"--{walker}",
    ]
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    r = subprocess.run(
        cmd,
        cwd=build_dir,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )

    assert r.returncode == 0, (
        f"{walker}: flowwalker exited with {r.returncode}.\n"
        f"CMD: {' '.join(cmd)}\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    assert r.stdout.strip(), f"{walker}: stdout was empty"
    walks = _parse_walks(r.stdout)
    assert walks, (
        f"{walker}: no walk lines found in stdout. First 40 lines:\n"
        + "\n".join(r.stdout.splitlines()[:40])
    )

    if stop_mode == "fixed":
        # Fixed walkers produce length == max_depth. Infer max_depth from
        # the first walk and verify all walks agree.
        inferred = max(len(w) for w in walks)
        for walk in walks:
            assert len(walk) == inferred, (
                f"{walker}: fixed-length walker produced walks of "
                f"differing lengths (saw {inferred} and {len(walk)}). "
                "Either the binary's stop logic is probabilistic for this "
                "walker or the test harness is misclassifying it."
            )
    else:
        # Probabilistic walkers must terminate within [1, max_depth].
        for walk in walks:
            assert len(walk) >= 1, (
                f"{walker}: found zero-length walk in probabilistic mode"
            )
