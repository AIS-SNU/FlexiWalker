"""Type analysis stage - extracts type and structure information."""

import json
from pathlib import Path
from typing import List, Dict, Any

from ..base import PipelineStage, PipelineError, ToolBuildHelper


class TypeExtractor(PipelineStage):
    """Extract type and structure information using AST analysis.
    
    This stage analyzes the walker template code using Clang AST to extract
    detailed type and structure information needed for code generation.
    """
    
    @property
    def name(self) -> str:
        return "type_analysis"
    
    @property
    def dependencies(self) -> List[Path]:
        return [
            self.config.get_artifact_path('walker_template'),
            self.config.get_artifact_path('llvm_analysis')
        ]
    
    @property
    def outputs(self) -> List[Path]:
        return [self.config.get_artifact_path('type_analysis')]
    
    def _tool(self) -> ToolBuildHelper:
        return ToolBuildHelper(
            stage=self,
            tool_dir=self.config.root_dir / "tools" / "clang" / "type-analyzer",
            executable_name="ast-tool",
        )

    def _run_ast_analysis(self):
        """Run AST analysis on the walker template.
        
        Executes the AST tool to analyze the walker template code and extract
        type and structure information for code generation.
        
        Raises:
            PipelineError: If analysis fails or required files are missing
        """
        tool = self._tool()
        template_file = self.config.get_artifact_path('walker_template')
        config_file = self.config.get_artifact_path('llvm_analysis')
        output_file = self.config.get_artifact_path('type_analysis')

        # Validate inputs exist
        if not template_file.exists():
            raise PipelineError(f"Walker template not found: {template_file}")
        if not config_file.exists():
            raise PipelineError(f"LLVM analysis config not found: {config_file}")
        if not tool.executable_path.exists():
            raise PipelineError(f"AST tool executable not found: {tool.executable_path}")

        cmd = [
            str(tool.executable_path),
            str(template_file),
            "-config", str(config_file),
            "-o", str(output_file),
            "--"
        ] + [str(flag) for flag in self.config.compile_flags]
        
        try:
            self.run_command(cmd)
            
            # Verify output was created
            if not output_file.exists():
                raise PipelineError(f"Type analysis file was not generated: {output_file}")
                
            self.logger.info(f"Type analysis completed: {output_file}")
            self.logger.debug(f"Analysis file size: {output_file.stat().st_size} bytes")
            
        except Exception as e:
            raise PipelineError(f"AST analysis failed: {e}") from e
    
    def execute(self):
        """Execute type analysis."""
        # Ensure artifacts directory exists
        self.config.ensure_artifacts_dir()
        
        # Build AST tool
        self._tool().ensure_built()

        # Run AST analysis
        self._run_ast_analysis()
        
        self.logger.info("Type analysis completed")
    
    def validate_output_content(self):
        """Validate that the type analysis produced valid output.
        
        Checks that the type analysis file exists and contains
        expected type and structure information.
        """
        super().validate_output_content()
        
        output_file = self.config.get_artifact_path('type_analysis')
        if not output_file.exists():
            return
        
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                self.logger.warning(f"Type analysis should be a dictionary, got {type(data)}")
                return
            
            # Check for expected analysis content
            # Type analysis typically contains class-to-field mappings
            if len(data) == 0:
                self.logger.warning("Type analysis file appears to be empty")
                return
            
            # Validate structure - should have walker classes with field information
            valid_entries = 0
            for class_name, class_data in data.items():
                if isinstance(class_data, dict) and len(class_data) > 0:
                    valid_entries += 1
            
            if valid_entries == 0:
                self.logger.warning("Type analysis contains no valid class data")
            else:
                self.logger.debug(f"Type analysis validation passed - {valid_entries} classes analyzed")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Type analysis file contains invalid JSON: {e}")
        except Exception as e:
            self.logger.warning(f"Could not validate type analysis file: {e}")
