"""
Reference-parameter handling in the dummy stub: a user method with a
`const T&` parameter must produce a well-formed call in walker_dummy.cu.
The metadata extractor's getMethodArgValue handles pointers, integers,
and floats — references would previously fall through to the fallback
literal "0", which fails to compile when bound to a struct reference.

This test pins that the pipeline runs cleanly when a walker exposes a
const-ref user method.
"""

from pathlib import Path

import pytest

from tests.conftest import run_pipeline_clean


pytestmark = pytest.mark.compiler


def test_const_ref_user_method_pipeline_succeeds(pipeline_workspace):
    repo_root = pipeline_workspace["repo_root"]
    app_cuh = repo_root / "include" / "app.cuh"
    fixture = repo_root / "tests" / "compiler" / "fixtures" / "user_method_const_ref.cuh"

    original = app_cuh.read_text()
    fixture_text = fixture.read_text()
    app_cuh.write_text(
        original.rstrip()
        + "\n\n// === appended by test_user_method_const_ref.py ===\n"
        + fixture_text
    )

    try:
        result = run_pipeline_clean(repo_root)
        if result.returncode != 0:
            pytest.fail(
                f"Pipeline failed (exit {result.returncode}) with const-ref "
                f"user method.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
    finally:
        app_cuh.write_text(original)
