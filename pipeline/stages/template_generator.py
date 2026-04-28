"""Template generation stage - generates walker template code."""

import json
from pathlib import Path
from typing import List, Dict, Any
from jinja2 import Environment, TemplateNotFound

from ..base import make_jinja_env

from ..base import PipelineStage, PipelineError


class TemplateGenerator(PipelineStage):
    """Generate walker template code from metadata.
    
    This stage takes the walker metadata extracted in the previous stage
    and generates template code that will be used for LLVM analysis.
    """
    
    @property
    def name(self) -> str:
        return "template_generation"
    
    @property
    def dependencies(self) -> List[Path]:
        return [self.config.get_artifact_path('walker_metadata')]
    
    @property
    def outputs(self) -> List[Path]:
        return [self.config.get_artifact_path('walker_template')]
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load walker metadata from JSON.
        
        Returns:
            Dictionary containing walker metadata
            
        Raises:
            PipelineError: If metadata file is missing or invalid
        """
        metadata_file = self.config.get_artifact_path('walker_metadata')
        
        if not metadata_file.exists():
            raise PipelineError(f"Metadata file not found: {metadata_file}")
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if not isinstance(metadata, dict):
                raise PipelineError(f"Metadata should be a dictionary, got {type(metadata)}")
            
            self.logger.debug(f"Loaded metadata with {len(metadata)} walker classes")
            return metadata
            
        except json.JSONDecodeError as e:
            raise PipelineError(f"Invalid JSON in metadata file: {e}") from e
        except Exception as e:
            raise PipelineError(f"Failed to load metadata: {e}") from e
    
    def _setup_jinja_env(self) -> Environment:
        """Setup Jinja2 environment for pre-analysis templates."""
        return make_jinja_env("pre")
    
    def _generate_walker_template(self, metadata: Dict[str, Any]):
        """Generate walker template code.
        
        Args:
            metadata: Dictionary containing walker class metadata
            
        Raises:
            PipelineError: If template generation fails
        """
        try:
            env = self._setup_jinja_env()
            
            # Load the walker dummy template
            try:
                template = env.get_template("walker_dummy.j2")
            except TemplateNotFound as e:
                raise PipelineError(f"Template file not found: walker_dummy.j2") from e
            
            # Render the template with walker metadata
            rendered = template.render(walkers=metadata)
            
            # Validate rendered content
            if not rendered.strip():
                raise PipelineError("Template rendering produced empty output")
            
            # Write to output file
            output_file = self.config.get_artifact_path('walker_template')
            with open(output_file, 'w') as f:
                f.write(rendered)
            
            # Verify output file was created
            if not output_file.exists():
                raise PipelineError(f"Failed to create output file: {output_file}")
            
            self.logger.info(f"Generated walker template: {output_file}")
            self.logger.debug(f"Template size: {output_file.stat().st_size} bytes")
            
        except Exception as e:
            if isinstance(e, PipelineError):
                raise
            raise PipelineError(f"Template generation failed: {e}") from e
    
    def execute(self):
        """Execute template generation."""
        # Ensure artifacts directory exists
        self.config.ensure_artifacts_dir()
        
        # Load metadata
        metadata = self._load_metadata()
        
        # Generate walker template
        self._generate_walker_template(metadata)
        
        self.logger.info("Template generation completed")
    
    def validate_output_content(self):
        """Validate that the template generation produced valid output.
        
        Checks that the generated template file exists and contains
        expected template code structure.
        """
        super().validate_output_content()
        
        output_file = self.config.get_artifact_path('walker_template')
        if not output_file.exists():
            return
        
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Basic validation - template should not be empty
            if not content.strip():
                self.logger.warning("Template file appears to be empty")
                return
            
            # The rendered template must contain the dummy_kernel
            # signature — the LLVM analyzer keys off that function in IR.
            # ('class' is not a meaningful marker here: the walker classes
            # live in the #include'd app.cuh, not the rendered text.)
            if 'dummy_kernel(' not in content:
                self.logger.warning("Template missing dummy_kernel signature")
            
            self.logger.debug(f"Template validation passed - {len(content)} characters generated")
                
        except Exception as e:
            self.logger.warning(f"Could not validate template file: {e}")
