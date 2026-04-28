"""
Code generation stage - generates final optimized implementation files.

This stage takes the analysis data from type analysis and LLVM analysis stages
and generates optimized CUDA implementation files using Jinja2 templates.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from jinja2 import Environment, TemplateError, TemplateNotFound

from ..base import make_jinja_env, load_pipeline_json

from ..base import PipelineStage, PipelineError


# Constants for code generation
class CodeGenConstants:
    """Constants used throughout code generation."""
    
    # File headers
    HEADER_GET_MAX_WEIGHT = "// Auto-generated get_max_weight method implementations"
    HEADER_GET_SUM_WEIGHT = "// Auto-generated get_sum_weight method implementations" 
    HEADER_FILL_DUMMY = "// Auto-generated fill_dummy functions"
    HEADER_GPU_GRAPH = "// Auto-generated unified struct extensions"
    HEADER_WALKER_TRAITS = "// Auto-generated WalkerTraits specializations based on class metadata"
    
    # Template names
    TEMPLATE_GET_MAX_WEIGHT = "get_max_weight.j2"
    TEMPLATE_GET_SUM_WEIGHT = "get_sum_weight.j2"
    TEMPLATE_FILL_DUMMY = "fill_dummy.j2"
    TEMPLATE_GPU_GRAPH = "gpu_graph.j2"
    
    # Special class names
    WALKER_META_CLASS = "WalkerMeta"
    
    # Field transformations
    ADJWGT_MAX_FIELD = "adjwgt_MAX"
    ADJWGT_MIN_FIELD = "adjwgt_MIN"
    ADJWGT_SUM_FIELD = "adjwgt_SUM"
    
    # Default field names
    DEFAULT_TASK_FIELDS = ["degree", "neighbor_offset"]


class CodeGenerator(PipelineStage):
    """
    Generate final optimized implementation files.
    
    This stage processes type analysis and LLVM analysis data to generate:
    - get_max_weight implementations for maximum weight calculations
    - get_sum_weight implementations for cumulative weight calculations
    - fill_dummy functions for dummy data generation
    - Extended GPU graph structures with optimized fields
    - Walker trait specializations based on analysis results
    """
    
    @property
    def name(self) -> str:
        return "code_generation"
    
    @property
    def dependencies(self) -> List[Path]:
        return [
            self.config.get_artifact_path('walker_metadata'),
            self.config.get_artifact_path('type_analysis'),
            self.config.get_artifact_path('llvm_analysis')
        ]
    
    @property
    def outputs(self) -> List[Path]:
        return [
            self.config.get_generated_path('gpu_graph'),
            self.config.get_generated_path('fill_dummy'),
            self.config.get_generated_path('get_max_weight'),
            self.config.get_generated_path('get_sum_weight'),
            self.config.get_generated_path('walker_traits')
        ]
    
    def _load_analysis_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load type analysis and LLVM analysis data from JSON files.

        Returns:
            Tuple of (type_data, llvm_data) dictionaries

        Raises:
            PipelineError: If files cannot be loaded or parsed
        """
        type_data = load_pipeline_json(self.config.get_artifact_path('type_analysis'))
        llvm_data = load_pipeline_json(self.config.get_artifact_path('llvm_analysis'))
        self.logger.debug(f"Loaded type analysis with {len(type_data)} entries")
        self.logger.debug(f"Loaded LLVM analysis with {len(llvm_data)} entries")
        return type_data, llvm_data

    def _validate_upstream_artifacts(
        self,
        walker_metadata: Dict[str, Any],
        type_analysis: Dict[str, Any],
    ) -> None:
        """
        Validate that upstream stages produced usable output for every walker.

        Raises PipelineError with an actionable message if:
          - a walker declared in walker_metadata has no type-analysis entry
          - a walker's branches list is missing or empty
          - a walker requests a (field, role) pair that graph_fields.config
            doesn't declare via preprocess=... (bug #1 / bug #5)

        Spec: §3.5 (bug #5).
        """
        walker_names = [name for name in walker_metadata.keys() if name != "headers"]
        for name in walker_names:
            if name not in type_analysis:
                raise PipelineError(
                    f"no type-analysis data for '{name}'. "
                    f"Check tools/clang/type-analyzer logs for failures."
                )
            entry = type_analysis[name]
            if not isinstance(entry, dict):
                raise PipelineError(
                    f"type_analysis for '{name}' is not an object "
                    f"(got {type(entry).__name__}). Pipeline output corrupted."
                )
            branches = entry.get("branches")
            if branches is None:
                raise PipelineError(
                    f"type_analysis for '{name}' missing 'branches' key. "
                    f"Likely pipeline output corruption — regenerate artifacts."
                )
            if not branches:
                raise PipelineError(
                    f"no return branches for '{name}' — likely a malformed "
                    f"get_weight() in include/app.cuh (possibly no return "
                    f"statement, or the analyzer couldn't reach it)."
                )

        # Validate that every (field, role) pair the analyzer requested is
        # actually declared in graph_fields.json with a matching preprocess op.
        # get_sum_weight rewrites every MAX/MIN reference to _SUM, so any field
        # touched by get_weight also needs SUM declared.
        graph_fields_path = self.config.get_artifact_path('graph_fields')
        if not graph_fields_path.exists():
            return
        graph_fields = load_pipeline_json(graph_fields_path)
        declared = graph_fields.get("fields", {})
        for walker in walker_names:
            entry = type_analysis.get(walker, {})
            for branch in entry.get("branches", []):
                for fr in branch.get("field_roles", []):
                    field = fr.get("field")
                    role = (fr.get("role") or "").lower()
                    if not field or not role:
                        continue
                    # adjwgt is implicit; its ops are derived per-walker by
                    # _compute_implicit_adjwgt_ops, not declared in
                    # graph_fields.config, so skip the declared-ops check.
                    if field == "adjwgt":
                        continue
                    if field not in declared:
                        raise PipelineError(
                            f"walker '{walker}' uses field '{field}' in "
                            f"get_weight but it is not declared in "
                            f"graph_fields.config."
                        )
                    ops = [op.lower() for op in (declared[field].get("preprocess_ops") or [])]
                    if role not in ops:
                        raise PipelineError(
                            f"walker '{walker}' needs role '{role}' for field "
                            f"'{field}', but graph_fields.config declares "
                            f"preprocess={sorted(ops)}. Add '{role}' to the "
                            f"preprocess list."
                        )
                    if "sum" not in ops:
                        raise PipelineError(
                            f"walker '{walker}' uses field '{field}' "
                            f"(role '{role}'), but graph_fields.config "
                            f"declares preprocess={sorted(ops)} — 'sum' is "
                            f"required because get_sum_weight aggregates over "
                            f"edges. Add 'sum' to the preprocess list."
                        )
    
    def _setup_jinja_env(self) -> Environment:
        """Setup Jinja2 environment for post-analysis templates."""
        try:
            env = make_jinja_env("post")
            env.filters["prepend"] = lambda value, prefix: prefix + value
            env.filters["strip_ptr"] = lambda type_str: type_str.replace("*", "").strip()
            env.tests["endswith_max"] = lambda name: name.endswith("_MAX")
            return env
            
        except Exception as e:
            raise PipelineError(f"Failed to setup Jinja2 environment: {e}")
    
    def _deduplicate_preserve_order(self, seq: List[Any]) -> List[Any]:
        """
        Remove duplicates while preserving order.
        
        Args:
            seq: Input sequence that may contain duplicates
            
        Returns:
            List with duplicates removed, preserving original order
        """
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]
    
    def _clean_redundant_zero_lines(self, lines: List[str]) -> List[str]:
        """
        Clean up redundant zero lines from generated code.
        
        Removes mathematical operations with zero that don't affect the result
        and other redundant constructs that may arise during code generation.
        
        Args:
            lines: List of code lines to clean
            
        Returns:
            List of cleaned code lines
        """
        cleaned = []
        for line in lines:
            stripped = line.strip()
            
            # Remove line if it's just max with 0.0
            if re.fullmatch(r".*=\s*max\([^,]+,\s*0\.0\);", stripped):
                continue
            if re.fullmatch(r".*=\s*max\(0\.0,\s*[^)]+\);", stripped):
                continue
            
            # Remove "+ 0.0" or " + 0.0" fragments
            line = re.sub(r"\s*\+\s*0\.0", "", line)
            
            cleaned.append(line)
        
        return self._deduplicate_preserve_order(cleaned)
    
    def _load_template(self, env: Environment, template_name: str) -> Any:
        """
        Load a Jinja2 template with error handling.
        
        Args:
            env: Jinja2 environment
            template_name: Name of the template file
            
        Returns:
            Loaded template object
            
        Raises:
            PipelineError: If template cannot be loaded
        """
        try:
            return env.get_template(template_name)
        except TemplateNotFound as e:
            raise PipelineError(f"Template not found: {template_name}") from e
    
    def _write_file_header(self, file_handle, header_text: str, additional_headers: Optional[List[str]] = None) -> None:
        """
        Write standardized file header.
        
        Args:
            file_handle: Open file handle to write to
            header_text: Main header comment text
            additional_headers: Optional list of additional headers to include
        """
        file_handle.write(f"{header_text}\n\n")
        if additional_headers:
            file_handle.write("\n".join(additional_headers) + "\n\n")
    
    def _iter_walker_classes(self, data: Dict[str, Any]):
        """
        Yield ``(class_name, content)`` for every walker class in `data`,
        skipping the synthetic ``"headers"`` entry, the abstract
        ``WalkerMeta`` base, and any entry whose content isn't a dict.

        This is the canonical "what counts as a walker" filter; every
        loop over ``type_analysis.json`` items should go through it so
        that the four checks (headers / WalkerMeta / dict-shape / log)
        can't drift apart.
        """
        for class_name, content in data.items():
            if class_name == "headers":
                continue
            if class_name == CodeGenConstants.WALKER_META_CLASS:
                continue
            if not isinstance(content, dict):
                self.logger.warning(
                    f"Skipping {class_name!r}: expected dict, got "
                    f"{type(content).__name__}"
                )
                continue
            yield class_name, content


    def _generate_get_max_weight(self, data: Dict[str, Any], headers: List[str], env: Environment) -> None:
        """
        Generate get_max_weight implementations.
        
        Creates optimized weight calculation functions for different walker types
        based on analysis data.
        
        Args:
            data: Type analysis data containing branch information
            headers: Include headers for the generated file
            env: Jinja2 environment for template rendering
            
        Raises:
            PipelineError: If template rendering or file writing fails
        """
        template = self._load_template(env, CodeGenConstants.TEMPLATE_GET_MAX_WEIGHT)
        output_file = self.config.get_generated_path('get_max_weight')
        
        try:
            with open(output_file, "w") as f:
                self._write_file_header(f, CodeGenConstants.HEADER_GET_MAX_WEIGHT, headers)
                
                for class_name, content in self._iter_walker_classes(data):
                    # Process branches and expressions
                    branches = content.get("branches", [])
                    exprs = self._deduplicate_preserve_order([b["return_expr"] for b in branches])
                    exprs = [e for e in exprs if e.strip() != "0.0"]
                    
                    if not exprs:
                        continue
                    
                    initial_expr = exprs[0]
                    remaining_exprs = exprs[1:]
                    
                    # Process body lines
                    body_lines_raw = []
                    for branch in branches:
                        body_lines_raw.extend(branch.get("body", []))
                    
                    body_lines = self._clean_redundant_zero_lines(body_lines_raw)
                    
                    # Render and write function
                    try:
                        function_code = template.render(
                            class_name=class_name,
                            body=body_lines,
                            initial_expr=initial_expr,
                            remaining_exprs=remaining_exprs,
                        )
                        f.write(function_code + "\n\n")
                    except TemplateError as e:
                        raise PipelineError(f"Template rendering failed for {class_name}: {e}")
            
            self.logger.info(f"Generated get_max_weight: {output_file}")
            
        except IOError as e:
            raise PipelineError(f"Failed to write get_max_weight file: {e}")
    
    def _generate_get_sum_weight(self, data: Dict[str, Any], headers: List[str], env: Environment) -> None:
        """
        Generate get_sum_weight implementations.
        
        Creates optimized cumulative weight calculation functions by transforming
        MAX operations to SUM operations and handling appropriate scaling.
        
        Args:
            data: Type analysis data containing branch information
            headers: Include headers for the generated file
            env: Jinja2 environment for template rendering
            
        Raises:
            PipelineError: If template rendering or file writing fails
        """
        template = self._load_template(env, CodeGenConstants.TEMPLATE_GET_SUM_WEIGHT)
        output_file = self.config.get_generated_path('get_sum_weight')
        
        try:
            with open(output_file, "w") as f:
                self._write_file_header(f, CodeGenConstants.HEADER_GET_SUM_WEIGHT, headers)
                
                for class_name, content in self._iter_walker_classes(data):
                    # Process branches and expressions
                    branches = content.get("branches", [])
                    all_exprs_raw = self._deduplicate_preserve_order([b["return_expr"] for b in branches])
                    
                    if not all_exprs_raw:
                        continue
                    
                    num_terms = len(all_exprs_raw)
                    # Generic SUM-rewrite: every preprocessed edge field carries a
                    # `_MAX` / `_MIN` suffix emitted by the monotonicity rewriter,
                    # always immediately followed by `[idx]`. get_sum_weight sums
                    # over edges, so both roles collapse to `_SUM`. Anchoring on
                    # the `[` avoids clobbering unrelated identifiers that happen
                    # to end in `_MAX` / `_MIN` (e.g. FLT_MAX).
                    # `_validate_upstream_artifacts` checks that every walker's
                    # used fields have SUM declared.
                    def _to_sum(s: str) -> str:
                        return re.sub(r'\b(\w+)_(MAX|MIN)\[', r'\1_SUM[', s)
                    exprs = [_to_sum(e) for e in all_exprs_raw if e.strip() != "0.0"]

                    if not exprs:
                        continue

                    initial_expr = exprs[0]
                    remaining_exprs = exprs[1:]

                    # Process body lines with field replacement
                    body_lines_raw = []
                    for branch in branches:
                        for line in branch.get("body", []):
                            body_lines_raw.append(_to_sum(line))
                    
                    body_lines = self._clean_redundant_zero_lines(body_lines_raw)
                    
                    uses_sum_field = any(CodeGenConstants.ADJWGT_SUM_FIELD in e for e in exprs) or \
                                   any(CodeGenConstants.ADJWGT_SUM_FIELD in line for line in body_lines_raw)
                    
                    # Render and write function
                    try:
                        function_code = template.render(
                            class_name=class_name,
                            body=body_lines,
                            initial_expr=initial_expr,
                            remaining_exprs=remaining_exprs,
                            num_terms=num_terms,
                            needs_division=(num_terms > 1),
                            multiply_by_degree=(not uses_sum_field)
                        )
                        f.write(function_code + "\n\n")
                    except TemplateError as e:
                        raise PipelineError(f"Template rendering failed for {class_name}: {e}")
            
            self.logger.info(f"Generated get_sum_weight: {output_file}")
            
        except IOError as e:
            raise PipelineError(f"Failed to write get_sum_weight file: {e}")
    
    def _generate_fill_dummy(self, data: Dict[str, Any], headers: List[str], env: Environment) -> None:
        """
        Generate fill_dummy implementations.
        
        Creates functions to fill dummy data structures for testing and initialization.
        
        Args:
            data: Type analysis data containing task field information
            headers: Include headers for the generated file
            env: Jinja2 environment for template rendering
            
        Raises:
            PipelineError: If template rendering or file writing fails
        """
        template = self._load_template(env, CodeGenConstants.TEMPLATE_FILL_DUMMY)
        output_file = self.config.get_generated_path('fill_dummy')
        
        try:
            with open(output_file, "w") as f:
                self._write_file_header(f, CodeGenConstants.HEADER_FILL_DUMMY, headers)
                
                for class_name, content in self._iter_walker_classes(data):
                    # Build field list with defaults
                    task_fields = content.get("task_fields", [])
                    all_fields = CodeGenConstants.DEFAULT_TASK_FIELDS.copy()
                    for field in task_fields:
                        if field not in all_fields:
                            all_fields.append(field)
                    
                    # Render and write function
                    try:
                        function_code = template.render(
                            class_name=class_name,
                            all_fields=all_fields
                        )
                        f.write(function_code + "\n\n")
                    except TemplateError as e:
                        raise PipelineError(f"Template rendering failed for {class_name}: {e}")
            
            self.logger.info(f"Generated fill_dummy: {output_file}")
            
        except IOError as e:
            raise PipelineError(f"Failed to write fill_dummy file: {e}")
    
    def _extract_struct_fields(self, data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Tuple[str, List[str]]]], set]:
        """
        Extract struct field information from analysis data.

        Args:
            data: Type analysis data

        Returns:
            Tuple of (struct_fields_map, all_class_names_set)
        """
        # Map of struct_name -> { field_name -> (field_type, [class_names]) }
        struct_fields = {}
        all_class_names = set()

        for class_name, content in self._iter_walker_classes(data):
            # Add ALL walker classes to the set (not just those that access fields)
            all_class_names.add(class_name)

            accessed_names = content.get("accessedFieldNames", {})
            accessed_types = content.get("accessedFieldTypes", {})

            for struct_name, fields in accessed_names.items():
                type_key = struct_name if struct_name in accessed_types else f"class.{struct_name}"
                types = accessed_types.get(type_key, [])

                if len(fields) != len(types):
                    continue  # Skip mismatched entries

                for fname, ftype in zip(fields, types):
                    entry = struct_fields.setdefault(struct_name, {}).setdefault(fname, (ftype, []))
                    entry[1].append(class_name)

        return struct_fields, all_class_names
    
    # Pattern for adjwgt accesses in body/return strings emitted by the
    # MonotonicityRewriter / ReturnVisitor. Anchored on `[` so unrelated
    # identifiers ending in _MAX/_MIN (e.g. FLT_MAX) don't match.
    _ADJWGT_OP_PATTERN = re.compile(r"\badjwgt_(MAX|MIN|SUM)\[")

    def _compute_implicit_adjwgt_ops(self, type_data: Dict[str, Any]) -> List[str]:
        """
        Return the preprocess ops actually needed for the implicit `adjwgt`
        field, derived from what the type analyzer emitted into the per-walker
        body / return_expr strings.

        The previous hardcoded ``["max", "min", "sum"]`` over-allocated:
        production walkers like Node2vec_weighted only ever read
        ``adjwgt_MAX[...]``, but the codegen was still asking the runtime to
        materialize ``adjwgt_MIN`` (an extra |E| floats per edge field).
        get_sum_weight rewrites every ``_MAX`` / ``_MIN`` reference to ``_SUM``,
        so ``sum`` is required whenever adjwgt is used at all.
        """
        ops: set = set()
        any_adjwgt = False
        for _class_name, content in self._iter_walker_classes(type_data):
            for branch in content.get("branches", []):
                strings = [branch.get("return_expr", "") or ""]
                strings.extend(branch.get("body", []) or [])
                for s in strings:
                    for m in self._ADJWGT_OP_PATTERN.finditer(s):
                        ops.add(m.group(1).lower())
                        any_adjwgt = True
        if any_adjwgt:
            ops.add("sum")
        return sorted(ops)

    def _preprocess_ops_for(
        self,
        field_name: str,
        graph_fields: Dict[str, Any],
        implicit_adjwgt_ops: List[str],
    ) -> List[str]:
        """
        Return the declared preprocess ops (sorted) for a given edge field.

        For the implicit `adjwgt` field (not present in graph_fields.config),
        returns the role-derived list computed by
        :meth:`_compute_implicit_adjwgt_ops`. For declared edge fields, returns
        the user-specified `preprocess_ops`. Returns [] for declared fields
        with no preprocessing requested.
        """
        explicit = graph_fields.get(field_name)
        if explicit is None:
            if field_name == "adjwgt":
                return list(implicit_adjwgt_ops)
            return []
        ops = explicit.get("preprocess_ops") or []
        return sorted(ops)

    def _build_extended_fields(
        self,
        fields: Dict[str, Tuple[str, List[str]]],
        graph_fields: Dict[str, Any],
        implicit_adjwgt_ops: List[str],
    ) -> List[Tuple[str, str, str]]:
        """
        Build per-op extended field entries for declaration and cleanup.

        For each accessed edge field, emit one (type, `{name}_{OP}`, condition)
        tuple per declared preprocess op (MAX/MIN/SUM).

        Args:
            fields: Field name -> (field_type, class_list) mapping
            graph_fields: Loaded graph_fields.json "fields" mapping
            implicit_adjwgt_ops: Ops needed for the implicit adjwgt field, as
                computed by :meth:`_compute_implicit_adjwgt_ops`.

        Returns:
            List of (field_type, field_name_with_op, condition_expr) tuples
        """
        extended_fields = []

        for fname, (ftype, class_list) in fields.items():
            ops = self._preprocess_ops_for(fname, graph_fields, implicit_adjwgt_ops)
            if not ops:
                continue

            condition_expr = " || ".join(f"FLAGS_{cls}" for cls in class_list)
            for op in ops:
                extended_fields.append(
                    (ftype, f"{fname}_{op.upper()}", condition_expr)
                )

        return extended_fields

    def _build_edge_preprocess_fields(
        self,
        fields: Dict[str, Tuple[str, List[str]]],
        graph_fields: Dict[str, Any],
        implicit_adjwgt_ops: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Build grouped per-field context for the preprocess-kernel invocation
        loop. Each entry carries the base pointer name, data type, the declared
        preprocess ops, and the walker-flag condition.
        """
        edge_preprocess_fields: List[Dict[str, Any]] = []

        for fname, (ftype, class_list) in fields.items():
            ops = self._preprocess_ops_for(fname, graph_fields, implicit_adjwgt_ops)
            if not ops:
                continue

            edge_preprocess_fields.append({
                "base_name": fname,
                "data_type": ftype.replace("*", ""),
                "preprocess_ops": ops,
                "condition": " || ".join(f"FLAGS_{cls}" for cls in class_list),
            })

        return edge_preprocess_fields
    
    def _write_gpu_graph_headers(self, file_handle, all_class_names: set) -> None:
        """
        Write GPU graph file headers and declarations.
        
        Args:
            file_handle: Open file handle
            all_class_names: Set of all class names for flag declarations
        """
        file_handle.write("#pragma once\n")
        file_handle.write("#include <gflags/gflags.h>\n")
        file_handle.write("#include <algorithm>\n")
        file_handle.write("#include <iostream>\n")
        file_handle.write("#include <limits>\n")
        file_handle.write("#include \"util.cuh\"\n")
        file_handle.write("#include \"gpu_graph_base.cuh\"\n")
        file_handle.write("#include \"generated/graph.cuh\"\n\n")
        file_handle.write(f"{CodeGenConstants.HEADER_GPU_GRAPH}\n\n")

        # Emit DECLARE_bool() lines at the top
        for cls in sorted(all_class_names):
            file_handle.write(f"DECLARE_bool({cls});\n")
        file_handle.write("\n")
    
    def _generate_graph(self, all_class_names: set) -> None:
        """
        Generate graph class with runtime-configurable field loading.

        Creates extended graph class that loads additional fields from config.

        Args:
            all_class_names: Set of all walker class names for flag declarations

        Raises:
            PipelineError: If template rendering or file writing fails
        """
        # Load graph fields configuration
        graph_fields_file = self.config.get_artifact_path('graph_fields')

        try:
            if not graph_fields_file.exists():
                self.logger.warning(f"Graph fields config not found: {graph_fields_file}")
                # Create empty graph class
                fields = {}
            else:
                with open(graph_fields_file, 'r') as f:
                    graph_fields_data = json.load(f)
                fields = graph_fields_data.get('fields', {})
        except Exception as e:
            raise PipelineError(f"Failed to load graph fields config: {e}")

        # Setup Jinja environment with custom filter.
        # graph.j2 relies on raw whitespace, so disable trim/lstrip.
        env = make_jinja_env("post", trim_blocks=False, lstrip_blocks=False)

        # Add custom filter for flag prefixing
        def add_flags_prefix(walker_name):
            return f"FLAGS_{walker_name}"
        env.filters['add_flags_prefix'] = add_flags_prefix

        template = env.get_template("graph.j2")
        output_file = self.config.get_generated_path('graph')

        try:
            with open(output_file, "w") as f:
                rendered = template.render(
                    all_walkers=sorted(all_class_names),
                    fields=fields
                )
                f.write(rendered)

            self.logger.info(f"Generated graph: {output_file}")

        except IOError as e:
            raise PipelineError(f"Failed to write graph file: {e}")
        except TemplateError as e:
            raise PipelineError(f"Template rendering failed for graph: {e}")

    def _load_graph_fields_for_gpu(self) -> Dict[str, Any]:
        """Load additional fields from graph_fields.json for GPU allocation."""
        graph_fields_file = self.config.get_artifact_path('graph_fields')

        try:
            if not graph_fields_file.exists():
                return {}

            with open(graph_fields_file, 'r') as f:
                graph_fields_data = json.load(f)
            return graph_fields_data.get('fields', {})
        except Exception as e:
            self.logger.warning(f"Failed to load graph fields for GPU: {e}")
            return {}

    def _generate_gpu_graph(self, data: Dict[str, Any], env: Environment) -> None:
        """
        Generate gpu_graph class extensions.

        Creates extended GPU graph structures with optimized fields based on
        walker analysis results and field access patterns, plus additional
        fields from graph_fields.json (edge_label, etc.).

        Args:
            data: Type analysis data containing field access information
            env: Jinja2 environment for template rendering

        Raises:
            PipelineError: If template rendering or file writing fails
        """
        # Load additional fields configuration
        additional_fields = self._load_graph_fields_for_gpu()

        # Setup custom Jinja environment with filters and tests.
        # gpu_graph.j2 relies on raw whitespace, so disable trim/lstrip.
        custom_env = make_jinja_env("post", trim_blocks=False, lstrip_blocks=False)

        # Add custom filters
        def add_flags_prefix(walker_name):
            return f"FLAGS_{walker_name}"
        custom_env.filters['add_flags_prefix'] = add_flags_prefix

        # Add custom tests (used with 'is' keyword in templates)
        def endswith_max(name):
            return name.endswith("_MAX")
        custom_env.tests['endswith_max'] = endswith_max

        template = custom_env.get_template("gpu_graph.j2")
        output_file = self.config.get_generated_path('gpu_graph')

        try:
            # Extract struct fields and class names
            struct_fields, all_class_names = self._extract_struct_fields(data)
            implicit_adjwgt_ops = self._compute_implicit_adjwgt_ops(data)

            with open(output_file, "w") as f:
                # Write headers and declarations
                self._write_gpu_graph_headers(f, all_class_names)

                # Generate struct extensions
                for struct_name, fields in struct_fields.items():
                    extended_fields = self._build_extended_fields(
                        fields, additional_fields, implicit_adjwgt_ops
                    )
                    edge_preprocess_fields = self._build_edge_preprocess_fields(
                        fields, additional_fields, implicit_adjwgt_ops
                    )

                    try:
                        rendered = template.render(
                            struct_name="gpu_graph" if struct_name == "gpu_graph_base" else struct_name,
                            extra_fields=extended_fields,
                            edge_preprocess_fields=edge_preprocess_fields,
                            additional_fields=additional_fields,
                        )
                        f.write(rendered + "\n\n")
                    except TemplateError as e:
                        raise PipelineError(f"Template rendering failed for {struct_name}: {e}")

            self.logger.info(f"Generated gpu_graph: {output_file}")

        except IOError as e:
            raise PipelineError(f"Failed to write gpu_graph file: {e}")
    
    def _generate_traits(
        self,
        llvm_data: Dict[str, Any],
        type_data: Dict[str, Any],
    ) -> None:
        """
        Generate walker traits.

        Creates specialized trait templates based on LLVM analysis results
        to optimize walker behavior at compile time.

        Args:
            llvm_data: LLVM analysis data containing walker characteristics
            type_data: Type analysis data containing per-branch force_eRVS_only

        Raises:
            PipelineError: If file writing fails
        """
        output_file = self.config.get_generated_path('walker_traits')

        try:
            with open(output_file, "w") as f:
                f.write(f"{CodeGenConstants.HEADER_WALKER_TRAITS}\n\n")
                f.write("#include \"app.cuh\"\n\n")

                # Primary template declaration
                f.write("template <typename T>\n")
                f.write("struct WalkerTraits;\n\n")

                # Generate specializations
                for class_name, content in self._iter_walker_classes(llvm_data):
                    possible_zero = content.get("possibleZero", 0)
                    update_flag = content.get("updateFlag", 0)
                    # Phase 2: the monotonicity rewriter can force eRVS_only
                    # independently of the LLVM complexity analyzer. Merge both.
                    llvm_ervs = bool(content.get("eRVS_only", False))
                    branches = type_data.get(class_name, {}).get("branches", [])
                    mono_ervs = any(b.get("force_eRVS_only", False) for b in branches)
                    ervs_only = 1 if (llvm_ervs or mono_ervs) else 0

                    f.write(f"template <>\n")
                    f.write(f"struct WalkerTraits<{class_name}> {{\n")
                    f.write(f"    static constexpr bool POSSIBLE_ZERO = {possible_zero};\n")
                    f.write(f"    static constexpr char UPDATE_FLAG = {update_flag};\n")
                    f.write(f"    static constexpr char ERVS_ONLY = {ervs_only};\n")
                    f.write("};\n\n")
            
            self.logger.info(f"Generated walker traits: {output_file}")
            
        except IOError as e:
            raise PipelineError(f"Failed to write walker traits file: {e}")
    
    def execute(self) -> None:
        """
        Execute code generation.
        
        Main execution method that coordinates the generation of all output files
        based on analysis data from previous pipeline stages.
        
        Raises:
            PipelineError: If any stage of code generation fails
        """
        try:
            # Ensure generated directory exists
            self.config.ensure_generated_dir()
            
            # Load analysis data
            type_data, llvm_data = self._load_analysis_data()

            # Load walker metadata and validate upstream artifacts before
            # any rendering. This surfaces actionable errors early (bug #5).
            walker_metadata = load_pipeline_json(
                self.config.get_artifact_path('walker_metadata')
            )
            self._validate_upstream_artifacts(walker_metadata, type_data)

            # Extract headers and data
            headers = type_data.get("headers", [])
            data = {k: v for k, v in type_data.items() if k != "headers"}

            # Validate we have data to process
            if not data:
                self.logger.warning("No type analysis data found to process")
                return
            
            # Setup Jinja environment
            env = self._setup_jinja_env()

            # Extract all class names first (needed for graph generation)
            _, all_class_names = self._extract_struct_fields(data)

            # Generate all output files
            self.logger.info("Starting code generation for all output files")
            self._generate_graph(all_class_names)  # Generate graph first (needed by gpu_graph)
            self._generate_get_max_weight(data, headers, env)
            self._generate_get_sum_weight(data, headers, env)
            self._generate_fill_dummy(data, headers, env)
            self._generate_gpu_graph(data, env)
            self._generate_traits(llvm_data, data)
            
            # Validate all outputs were created
            missing_outputs = []
            for output_path in self.outputs:
                if not output_path.exists():
                    missing_outputs.append(str(output_path))
            
            if missing_outputs:
                raise PipelineError(f"Failed to generate output files: {missing_outputs}")
            
            self.logger.info("Code generation completed successfully")
            
        except Exception as e:
            if isinstance(e, PipelineError):
                raise
            raise PipelineError(f"Code generation failed: {e}")
