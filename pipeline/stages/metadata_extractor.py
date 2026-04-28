"""Metadata extraction stage - extracts walker class information from source code."""

import json
from pathlib import Path
from typing import List, Dict, Any

from ..base import PipelineStage, PipelineError, ToolBuildHelper


class MetadataExtractor(PipelineStage):
    """Extract walker metadata from source code using Clang AST.
    
    This stage analyzes the walker source code to extract metadata about
    walker classes, their methods, and structure. This metadata is used
    by subsequent stages for code generation.
    """
    
    @property
    def name(self) -> str:
        return "metadata_extraction"
    
    @property
    def dependencies(self) -> List[Path]:
        """Depends on source code and generated headers."""
        return [
            self.config.include_dir / "app.cuh",
            # We also depend on dummy headers being available
        ]
    
    @property
    def outputs(self) -> List[Path]:
        return [self.config.get_artifact_path('walker_metadata')]
    
    def _tool(self) -> ToolBuildHelper:
        return ToolBuildHelper(
            stage=self,
            tool_dir=self.config.root_dir / "tools" / "clang" / "metadata-extractor",
            executable_name="pre-tool",
        )

    def _create_dummy_entry(self) -> Path:
        """Create dummy entry file for analysis.
        
        This file includes the main app header and serves as an entry point
        for the Clang AST analysis tool.
        
        Returns:
            Path to the created dummy file
        """
        dummy_file = self.config.artifacts_dir / "dummy_entry.cu"
        
        try:
            with open(dummy_file, "w") as f:
                f.write('#include "app.cuh"\n')
            
            self.logger.debug(f"Created dummy entry file: {dummy_file}")
            return dummy_file
            
        except Exception as e:
            raise PipelineError(f"Failed to create dummy entry file: {e}") from e
    
    def _run_analysis(self, dummy_file: Path):
        """Run the metadata extraction analysis.
        
        Executes the Clang-based tool to analyze the walker source code
        and extract metadata about classes and their structure.
        
        Args:
            dummy_file: Path to the dummy entry file for analysis
        """
        tool = self._tool()
        output_file = self.config.get_artifact_path('walker_metadata')

        # Validate inputs
        if not dummy_file.exists():
            raise PipelineError(f"Dummy entry file not found: {dummy_file}")
        if not tool.executable_path.exists():
            raise PipelineError(f"Tool executable not found: {tool.executable_path}")

        # Build command with compile flags
        cmd = [
            str(tool.executable_path),
            str(dummy_file),
            "-o", str(output_file),
            "--"
        ] + [str(flag) for flag in self.config.compile_flags]
        
        try:
            self.run_command(cmd)
            
            # Validate output was created
            if not output_file.exists():
                raise PipelineError(f"Tool did not generate expected output: {output_file}")
                
            self.logger.debug(f"Metadata analysis completed: {output_file}")
            
        except Exception as e:
            raise PipelineError(f"Metadata analysis failed: {e}") from e
    
    def execute(self):
        """Execute metadata extraction."""
        # Ensure artifacts directory exists
        self.config.ensure_artifacts_dir()

        # Create dummy entry file for Clang analysis
        dummy_file = self._create_dummy_entry()

        # Build the extraction tool
        self._tool().ensure_built()

        # Run the analysis
        self._run_analysis(dummy_file)

        self.logger.info(f"Metadata extracted to: {self.config.get_artifact_path('walker_metadata')}")

    def validate_output_content(self):
        """Validate that the metadata extraction produced valid output.
        
        Checks that the extracted metadata file exists and contains
        expected walker class information.
        """
        super().validate_output_content()
        
        output_file = self.config.get_artifact_path('walker_metadata')
        if not output_file.exists():
            return
        
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                self.logger.warning(f"Metadata should be a dictionary, got {type(data)}")
                return
            
            # Check for expected walker classes
            if len(data) == 0:
                self.logger.warning("Metadata file appears to be empty")
                return
            
            # Validate structure - metadata typically contains class information
            valid_classes = 0
            for class_name, class_info in data.items():
                if isinstance(class_info, dict):
                    valid_classes += 1
            
            if valid_classes == 0:
                self.logger.warning("Metadata contains no valid class information")
            else:
                self.logger.debug(f"Metadata validation passed - {valid_classes} classes found")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Metadata file contains invalid JSON: {e}")
        except Exception as e:
            self.logger.warning(f"Could not validate metadata file: {e}")
