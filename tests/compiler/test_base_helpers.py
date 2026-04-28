"""Unit tests for pipeline.base helpers added in Phase 2."""

import json
import tempfile
from pathlib import Path

import pytest

from pipeline.base import PipelineError, load_pipeline_json, make_jinja_env


pytestmark = pytest.mark.compiler


def test_load_pipeline_json_reads_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"a": 1, "b": [2, 3]}')
    data = load_pipeline_json(p)
    assert data == {"a": 1, "b": [2, 3]}


def test_load_pipeline_json_validates_required_keys(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"a": 1}')
    with pytest.raises(PipelineError, match="missing required key 'b'"):
        load_pipeline_json(p, required_keys=("a", "b"))


def test_load_pipeline_json_missing_file_raises(tmp_path):
    p = tmp_path / "nope.json"
    with pytest.raises(PipelineError, match="not found"):
        load_pipeline_json(p)


def test_load_pipeline_json_invalid_json_raises(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("not json")
    with pytest.raises(PipelineError, match="failed to parse"):
        load_pipeline_json(p)


def test_make_jinja_env_loads_from_subdir(tmp_path, monkeypatch, repo_root):
    env = make_jinja_env("post")
    assert env.trim_blocks is True
    assert env.lstrip_blocks is True
    tpl = env.get_template("get_max_weight.j2")
    assert tpl is not None
