"""Grammar tests for GraphFieldsConfigParser. Covers old + Phase-2 syntax."""

from pathlib import Path

import pytest

from pipeline.graph_config_parser import FieldDefinition, GraphFieldsConfigParser


pytestmark = pytest.mark.compiler


def _write_config(tmp_path, body: str) -> Path:
    p = tmp_path / "graph_fields.config"
    p.write_text(body)
    return p


def test_old_syntax_still_parses(tmp_path):
    """Existing 4-part syntax must parse unchanged — backcompat."""
    cfg = _write_config(
        tmp_path,
        "edge_label[Metapath,Metapath_weighted] = file:int:edge\n",
    )
    p = GraphFieldsConfigParser(str(cfg))
    assert p.is_loaded()
    assert p.has_field("edge_label")
    fd = p.get_field_definition("edge_label")
    assert fd.data_type == "int"
    assert fd.size_type == "edge"
    assert fd.preprocess_ops == set()


def test_new_preprocess_token_parses(tmp_path):
    cfg = _write_config(
        tmp_path,
        "edge_timestamp[MyWalker] = file:weight_t:edge:preprocess=max,min\n",
    )
    p = GraphFieldsConfigParser(str(cfg))
    assert p.is_loaded()
    fd = p.get_field_definition("edge_timestamp")
    assert fd.preprocess_ops == {"max", "min"}


def test_preprocess_sum_only(tmp_path):
    cfg = _write_config(
        tmp_path,
        "xx[MyWalker] = file:weight_t:edge:preprocess=sum\n",
    )
    fd = GraphFieldsConfigParser(str(cfg)).get_field_definition("xx")
    assert fd.preprocess_ops == {"sum"}


def test_preprocess_all_three(tmp_path):
    cfg = _write_config(
        tmp_path,
        "xx[MyWalker] = file:weight_t:edge:preprocess=max,min,sum\n",
    )
    fd = GraphFieldsConfigParser(str(cfg)).get_field_definition("xx")
    assert fd.preprocess_ops == {"max", "min", "sum"}


def test_preprocess_unknown_op_raises_or_warns(tmp_path):
    """Unknown ops must surface rather than be silently accepted."""
    cfg = _write_config(
        tmp_path,
        "xx[MyWalker] = file:weight_t:edge:preprocess=max,wiggle\n",
    )
    p = GraphFieldsConfigParser(str(cfg))
    assert not p.has_field("xx"), (
        "Fields with unknown preprocess ops must be rejected "
        "(skipped or raising)."
    )


def test_init_source_type_with_preprocess(tmp_path):
    """Init-source fields can also carry preprocess ops."""
    cfg = _write_config(
        tmp_path,
        "custom[MyWalker] = init:weight_t:edge:preprocess=max:1.0\n",
    )
    fd = GraphFieldsConfigParser(str(cfg)).get_field_definition("custom")
    assert fd.source_type == "init"
    assert fd.preprocess_ops == {"max"}
    assert fd.init_value == "1.0"
