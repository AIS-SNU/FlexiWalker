"""
Graph Field Detector Stage

Analyzes graph_fields.config to determine what additional fields need to be
generated in the graph and gpu_graph classes. Outputs configuration that will
be used by the code generator.
"""

import json
from pathlib import Path
from typing import List

from ..base import PipelineStage, PipelineError
from ..graph_config_parser import GraphFieldsConfigParser


class GraphFieldDetectorStage(PipelineStage):
    """Detects which graph fields are required by each walker from config."""

    @property
    def name(self) -> str:
        return "graph_field_detector"

    @property
    def dependencies(self) -> List[Path]:
        # graph_fields.config is optional - if it doesn't exist, we just generate empty fields
        return []

    @property
    def outputs(self) -> List[Path]:
        return [self.config.graph_fields_output]

    def execute(self):
        """Load field definitions and prepare for code generation."""
        self.logger.info("Reading graph field definitions...")

        # Ensure artifacts directory exists
        self.config.ensure_artifacts_dir()

        # Load field definitions from config/graph_fields.config
        config_parser = GraphFieldsConfigParser(str(self.config.graph_fields_config))

        if not config_parser.is_loaded():
            self.logger.info("No additional fields configured")
            # Still create output with empty fields
            output = {
                "fields": {},
                "walker_requirements": {}
            }
            with open(self.config.graph_fields_output, 'w') as f:
                json.dump(output, f, indent=2)
            self.logger.info(f"Created empty graph fields config: {self.config.graph_fields_output}")
            return

        # Build output structure
        fields = {}
        walker_requirements = {}

        for field_name in config_parser.get_all_fields():
            field_def = config_parser.get_field_definition(field_name)

            fields[field_name] = {
                "source_type": field_def.source_type,
                "data_type": field_def.data_type,
                "size_type": field_def.size_type,
                "init_value": field_def.init_value if field_def.source_type == "init" else "",
                "walkers": field_def.walkers,
                "preprocess_ops": sorted(field_def.preprocess_ops),
            }

            # Build reverse mapping: walker -> fields
            for walker in field_def.walkers:
                if walker not in walker_requirements:
                    walker_requirements[walker] = []
                walker_requirements[walker].append(field_name)

        # Output results
        output = {
            "fields": fields,
            "walker_requirements": walker_requirements
        }

        # Write output
        with open(self.config.graph_fields_output, 'w') as f:
            json.dump(output, f, indent=2)

        self.logger.info(f"Detected {len(fields)} additional fields:")
        for field_name, field_info in fields.items():
            walkers_str = ', '.join(field_info['walkers'])
            self.logger.info(f"  - {field_name} ({field_info['data_type']}) for {walkers_str}")

        self.logger.info(f"Generated graph fields config: {self.config.graph_fields_output}")
