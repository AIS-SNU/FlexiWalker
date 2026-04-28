"""Base classes and utilities for pipeline stages."""

import datetime
import json
import logging
import subprocess
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader

from .config import PipelineConfig


class PipelineError(Exception):
    """Exception raised for pipeline-related errors."""
    pass


class ToolBuildManager:
    """
    Manages incremental builds for compilation tools.

    This class provides smart caching for C++ tools built with CMake,
    only rebuilding when source files have actually changed.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def should_rebuild(self, tool_dir: Path, executable: Path) -> bool:
        """
        Check if tool needs rebuilding based on source file timestamps.

        Args:
            tool_dir: Directory containing tool source files
            executable: Path to the built executable/library

        Returns:
            True if rebuild is needed, False otherwise
        """
        # If executable doesn't exist, we must build
        if not executable.exists():
            self.logger.debug(f"Tool {executable.name} doesn't exist, rebuild needed")
            return True

        # Collect all source files that affect the build
        source_patterns = ["*.cpp", "*.h", "*.hpp", "*.cxx", "CMakeLists.txt"]
        source_files = []
        for pattern in source_patterns:
            source_files.extend(tool_dir.rglob(pattern))

        if not source_files:
            self.logger.warning(f"No source files found in {tool_dir}")
            return True

        # Compare timestamps
        exe_mtime = executable.stat().st_mtime
        for src in source_files:
            if src.exists() and src.stat().st_mtime > exe_mtime:
                self.logger.debug(f"Source file {src.name} newer than executable, rebuild needed")
                return True

        self.logger.debug(f"Tool {executable.name} is up-to-date")
        return False

    def build_tool(self, tool_dir: Path, build_dir: Path,
                   run_command_fn, ensure_dir_fn) -> None:
        """
        Build a CMake-based tool.

        Args:
            tool_dir: Directory containing the tool source
            build_dir: Directory for build artifacts
            run_command_fn: Function to run commands (from PipelineStage)
            ensure_dir_fn: Function to ensure directory exists (from PipelineStage)
        """
        # Clean and create build directory
        ensure_dir_fn(build_dir, clean=True)

        # Run CMake
        self.logger.info(f"Configuring build with CMake...")
        run_command_fn(["cmake", ".."], cwd=build_dir)

        # Build with make
        self.logger.info(f"Building with make...")
        run_command_fn(["make", "-j"], cwd=build_dir)

    def build_if_needed(self, tool_dir: Path, build_dir: Path,
                       executable: Path, run_command_fn, ensure_dir_fn) -> bool:
        """
        Build tool only if sources have changed.

        Args:
            tool_dir: Directory containing tool source files
            build_dir: Directory for build artifacts
            executable: Path to the built executable/library
            run_command_fn: Function to run commands (from PipelineStage)
            ensure_dir_fn: Function to ensure directory exists (from PipelineStage)

        Returns:
            True if build was performed, False if skipped
        """
        if not self.should_rebuild(tool_dir, executable):
            self.logger.info(f"Tool {executable.name} is up-to-date, skipping build")
            return False

        self.logger.info(f"Building {executable.name}...")
        self.build_tool(tool_dir, build_dir, run_command_fn, ensure_dir_fn)

        # Verify build succeeded
        if not executable.exists():
            raise PipelineError(f"Build failed - executable not found: {executable}")

        self.logger.info(f"Successfully built {executable.name}")
        return True


class PipelineStage(ABC):
    """Base class for all pipeline stages."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"pipeline.{self.name}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this stage."""
        pass
    
    @property
    @abstractmethod
    def dependencies(self) -> List[Path]:
        """Return list of required input files/directories."""
        pass
    
    @property
    @abstractmethod
    def outputs(self) -> List[Path]:
        """Return list of expected output files."""
        pass
    
    def validate_dependencies(self):
        """Check that all dependencies exist."""
        missing = []
        for dep in self.dependencies:
            if not dep.exists():
                missing.append(str(dep))
        
        if missing:
            raise PipelineError(
                f"Stage '{self.name}' missing dependencies: {', '.join(missing)}"
            )
    
    def validate_outputs(self):
        """Check that all expected outputs were created."""
        missing = []
        for output in self.outputs:
            if not output.exists():
                missing.append(str(output))
        
        if missing:
            raise PipelineError(
                f"Stage '{self.name}' failed to create outputs: {', '.join(missing)}"
            )
    
    def validate_output_content(self):
        """Validate that output files contain expected content. Override in subclasses."""
        # Default implementation - check files are not empty
        for output in self.outputs:
            if output.exists() and output.stat().st_size == 0:
                self.logger.warning(f"Output file is empty: {output}")
    
    def save_stage_metadata(self):
        """Save metadata about this stage's execution."""
        metadata = {
            "stage": self.name,
            "timestamp": datetime.datetime.now().isoformat(),
            "dependencies": [str(dep) for dep in self.dependencies],
            "outputs": [str(output) for output in self.outputs],
            "outputs_info": {}
        }
        
        # Collect output file info
        for output in self.outputs:
            if output.exists():
                metadata["outputs_info"][str(output)] = {
                    "size": output.stat().st_size,
                    "mtime": output.stat().st_mtime
                }
        
        metadata_file = self.config.artifacts_dir / f"{self.name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.debug(f"Saved stage metadata: {metadata_file}")
    
    def needs_rebuild(self) -> bool:
        """Check if stage needs to run based on file timestamps."""
        # If any output is missing, we need to rebuild
        if not all(output.exists() for output in self.outputs):
            return True
        
        # If no dependencies, always rebuild (e.g., first stage)
        if not self.dependencies:
            return True
        
        # Check if any dependency is newer than the oldest output
        try:
            oldest_output = min(output.stat().st_mtime for output in self.outputs)
            newest_dep = max(dep.stat().st_mtime for dep in self.dependencies if dep.exists())
            return newest_dep > oldest_output
        except (OSError, ValueError):
            # If we can't determine timestamps, rebuild to be safe
            return True
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, 
                   capture_output: bool = False) -> subprocess.CompletedProcess:
        """Run a command with proper logging and error handling."""
        cmd_str = ' '.join(str(c) for c in cmd)
        self.logger.info(f"Running: {cmd_str}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                capture_output=capture_output,
                text=True
            )
            self.logger.debug(f"Command succeeded: {cmd_str}")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {cmd_str}")
            if e.stdout:
                self.logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                self.logger.error(f"STDERR: {e.stderr}")
            raise PipelineError(f"Command failed: {cmd_str}") from e
    
    def ensure_dir(self, path: Path, clean: bool = False):
        """Ensure directory exists, optionally cleaning it first."""
        if clean and path.exists():
            self.logger.debug(f"Cleaning directory: {path}")
            shutil.rmtree(path)
        
        path.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Ensured directory exists: {path}")
    
    @abstractmethod
    def execute(self):
        """Execute the stage logic."""
        pass
    
    def run(self, force: bool = False):
        """Run the stage with dependency checking."""
        self.logger.info(f"Running stage: {self.name}")
        
        # Check dependencies
        self.validate_dependencies()
        
        # Check if rebuild is needed
        if not force and not self.needs_rebuild():
            self.logger.info(f"Stage '{self.name}' outputs are up-to-date, skipping")
            return
        
        # Execute the stage
        try:
            self.execute()
            self.validate_outputs()
            self.validate_output_content()
            self.save_stage_metadata()
            self.logger.info(f"Stage '{self.name}' completed successfully")
        except Exception as e:
            self.logger.error(f"Stage '{self.name}' failed: {e}")
            raise


def setup_logging(level: str = "INFO"):
    """Setup logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


class ToolBuildHelper:
    """
    Thin orchestrator around ToolBuildManager for the three C++ analyzer
    tools. Replaces the _get_tool_paths / _build_*_tool pairs previously
    duplicated in metadata_extractor, type_extractor, and llvm_analyzer.

    Usage from a PipelineStage:
        helper = ToolBuildHelper(
            stage=self,
            tool_dir=self.config.root_dir / "tools" / "clang" / "metadata-extractor",
            executable_name="pre-tool",
        )
        helper.ensure_built()
        exe = helper.executable_path
    """

    def __init__(self, stage, tool_dir: Path, executable_name: str):
        self.stage = stage
        self.tool_dir = Path(tool_dir)
        self.build_dir = self.tool_dir / "build"
        self.executable_name = executable_name
        self._manager = ToolBuildManager(stage.logger)

    @property
    def executable_path(self) -> Path:
        return self.build_dir / self.executable_name

    def ensure_built(self) -> None:
        """Build the tool if its sources are newer than the executable."""
        if not self.tool_dir.is_dir():
            raise PipelineError(f"tool source dir missing: {self.tool_dir}")
        self._manager.build_if_needed(
            tool_dir=self.tool_dir,
            build_dir=self.build_dir,
            executable=self.executable_path,
            run_command_fn=self.stage.run_command,
            ensure_dir_fn=self.stage.ensure_dir,
        )


def load_pipeline_json(
    path: Path,
    required_keys: Tuple[str, ...] = (),
) -> Dict:
    """
    Load a pipeline JSON artifact with standardized error handling.

    Raises PipelineError with an actionable message if the file is missing,
    malformed, has an unexpected top-level type, or violates the
    required-keys contract.
    """
    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"JSON artifact not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise PipelineError(f"failed to parse {path}: {e}") from e
    if not isinstance(data, dict):
        raise PipelineError(
            f"expected top-level object in {path}, got {type(data).__name__}"
        )
    for key in required_keys:
        if key not in data:
            raise PipelineError(f"{path}: missing required key '{key}'")
    return data


def make_jinja_env(
    template_subdir: str,
    trim_blocks: bool = True,
    lstrip_blocks: bool = True,
) -> Environment:
    """
    Construct a Jinja2 Environment rooted at tools/codegen/templates/<subdir>.
    Centralizing this avoids divergent per-stage Environment setup.

    trim_blocks/lstrip_blocks default to True (Python-friendly). Pass False
    for legacy templates that rely on raw whitespace (graph.j2, gpu_graph.j2).
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    template_root = repo_root / "tools" / "codegen" / "templates" / template_subdir
    if not template_root.is_dir():
        raise PipelineError(f"template directory missing: {template_root}")
    return Environment(
        loader=FileSystemLoader(str(template_root)),
        trim_blocks=trim_blocks,
        lstrip_blocks=lstrip_blocks,
    )
