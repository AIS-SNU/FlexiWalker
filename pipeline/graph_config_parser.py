"""
Config file parsers for graph field configuration.
Used by the pipeline to determine what fields to generate code for.

Two config files:
1. graph_fields.config - Field definitions (requires recompilation if changed)
2. graphs/<name>.config - Runtime paths (no recompilation needed)
"""

import logging

logger = logging.getLogger(__name__)


class FieldDefinition:
    """Definition of an additional graph field from graph_fields.config"""
    FILE_LOAD = "file"
    INIT_VALUE = "init"
    SIZE_EDGE = "edge"
    SIZE_NODE = "node"
    VALID_PREPROCESS_OPS = {"max", "min", "sum"}

    def __init__(self):
        self.source_type = self.FILE_LOAD
        self.data_type = ""       # e.g., "int", "weight_t", "float"
        self.size_type = self.SIZE_EDGE  # "edge" or "node"
        self.init_value = ""      # For INIT_VALUE source
        self.walkers = []         # List of walker names that need this field
        self.preprocess_ops = set()  # Subset of VALID_PREPROCESS_OPS

    def __repr__(self):
        if self.source_type == self.FILE_LOAD:
            return f"FieldDef(FILE, type={self.data_type}, size={self.size_type}, walkers={self.walkers}, preprocess={sorted(self.preprocess_ops)})"
        else:
            return f"FieldDef(INIT, type={self.data_type}, size={self.size_type}, value={self.init_value}, walkers={self.walkers}, preprocess={sorted(self.preprocess_ops)})"


class GraphFieldsConfigParser:
    """Parser for graph_fields.config - field definitions"""

    def __init__(self, config_file=None):
        self.fields = {}
        self.loaded = False
        if config_file:
            self.load(config_file)

    def load(self, config_file):
        """Load field definitions from graph_fields.config"""
        try:
            with open(config_file, 'r') as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue

                    # Parse "field_name[Walker1,Walker2] = source_type:data_type:init_value"
                    if '=' not in line:
                        logger.warning("Invalid config line %d: %s", line_num, line)
                        continue

                    left, right = line.split('=', 1)
                    left = left.strip()
                    right = right.strip()

                    # Parse field name and walkers
                    if '[' not in left or ']' not in left:
                        logger.warning("Missing walker list at line %d: %s", line_num, line)
                        continue

                    bracket_start = left.index('[')
                    bracket_end = left.index(']')
                    field_name = left[:bracket_start].strip()
                    walker_str = left[bracket_start+1:bracket_end].strip()
                    walkers = [w.strip() for w in walker_str.split(',')]

                    # Parse source specification: source_type:data_type:size_type:init_value
                    parts = right.split(':')
                    if len(parts) < 3:
                        logger.warning(
                            "Invalid source spec at line %d: %s "
                            "(expected source_type:data_type:size_type[:init_value])",
                            line_num, right,
                        )
                        continue

                    field_def = FieldDefinition()
                    field_def.source_type = parts[0].strip()
                    field_def.data_type = parts[1].strip()
                    field_def.size_type = parts[2].strip()
                    field_def.walkers = walkers

                    # Validate size_type
                    if field_def.size_type not in [FieldDefinition.SIZE_EDGE, FieldDefinition.SIZE_NODE]:
                        logger.warning(
                            "Invalid size_type '%s' at line %d (expected 'edge' or 'node')",
                            field_def.size_type, line_num,
                        )
                        continue

                    # Parse optional trailing tokens: preprocess=<csv> and init_value.
                    # Token order is flexible; init_value is the positional
                    # non-preprocess token, required for INIT_VALUE source type.
                    extras = [p.strip() for p in parts[3:]]
                    preprocess_ops = set()
                    init_value = ""
                    bad_preprocess = False
                    for token in extras:
                        if token.startswith("preprocess="):
                            raw = token[len("preprocess="):]
                            ops = {op.strip() for op in raw.split(",") if op.strip()}
                            invalid = ops - FieldDefinition.VALID_PREPROCESS_OPS
                            if invalid:
                                logger.warning(
                                    "unknown preprocess ops %s at line %d: %s. Skipping this field.",
                                    sorted(invalid), line_num, line,
                                )
                                bad_preprocess = True
                                break
                            preprocess_ops = ops
                        else:
                            init_value = token

                    if bad_preprocess:
                        continue

                    field_def.preprocess_ops = preprocess_ops

                    if field_def.source_type == FieldDefinition.INIT_VALUE:
                        if not init_value:
                            logger.warning("Missing init value at line %d: %s", line_num, right)
                            continue
                        field_def.init_value = init_value

                    self.fields[field_name] = field_def

            self.loaded = True
            logger.info("Loaded %d field definitions from %s", len(self.fields), config_file)
            return True

        except FileNotFoundError:
            logger.warning(
                "Field definitions file '%s' not found; no additional fields will be generated.",
                config_file,
            )
            return False
        except Exception as e:
            logger.error("Error loading field definitions: %s", e)
            return False

    def has_field(self, field_name):
        """Check if field is defined."""
        return field_name in self.fields

    def get_field_definition(self, field_name):
        """Get definition for a field."""
        return self.fields.get(field_name, None)

    def get_all_fields(self):
        """Get all defined field names."""
        return list(self.fields.keys())

    def get_walkers_for_field(self, field_name):
        """Get list of walkers that require this field."""
        field_def = self.fields.get(field_name)
        return field_def.walkers if field_def else []

    def is_loaded(self):
        """Check if config was successfully loaded."""
        return self.loaded


class GraphPathsConfigParser:
    """Parser for graphs/<name>.config - runtime file paths"""

    def __init__(self, config_file=None):
        self.field_paths = {}
        self.loaded = False
        if config_file:
            self.load(config_file)

    def load(self, config_file):
        """Load runtime paths from graph-specific config."""
        try:
            with open(config_file, 'r') as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue

                    # Parse "field_name = /path/to/file"
                    if '=' not in line:
                        logger.warning("Invalid config line %d: %s", line_num, line)
                        continue

                    field_name, path = line.split('=', 1)
                    field_name = field_name.strip()
                    path = path.strip()

                    if not field_name or not path:
                        logger.warning("Invalid config line %d: %s", line_num, line)
                        continue

                    self.field_paths[field_name] = path

            self.loaded = True
            return True

        except FileNotFoundError:
            logger.warning("Graph paths config file '%s' not found", config_file)
            return False
        except Exception as e:
            logger.error("Error loading graph paths config: %s", e)
            return False

    def get_path(self, field_name):
        """Get file path for a field."""
        return self.field_paths.get(field_name, "")

    def has_field(self, field_name):
        """Check if field has a configured path."""
        return field_name in self.field_paths

    def is_loaded(self):
        """Check if config was successfully loaded."""
        return self.loaded
