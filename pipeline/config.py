"""Pipeline configuration management."""

import os
from pathlib import Path


# Keep in sync with MAX_ANALYSIS_DEPTH / MAX_ANALYSIS_BRANCHES defaults
# inside tools/clang/type-analyzer/src/MonotonicityRewriter (Phase 2 spec §5.2.1.3).
DEFAULT_MAX_ANALYSIS_DEPTH = 6
DEFAULT_MAX_ANALYSIS_BRANCHES = 8


class PipelineConfig:
    """Centralized configuration for the FlexiWalker compilation pipeline."""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        
        # Core directories
        self.artifacts_dir = self.root_dir / "artifacts"
        self.include_dir = self.root_dir / "include"
        self.generated_dir = self.include_dir / "generated"
        self.tools_dir = self.root_dir / "tools"
        self.config_dir = self.root_dir / "config"
        self.graphs_config_dir = self.config_dir / "graphs"
        
        # Build configuration
        self.cuda_arch = "sm_86"
        self.llvm_version = "17"
        self.cuda_path = "/usr/local/cuda"

        # Complexity budgets for the type-analyzer's monotonicity walk.
        # Exceeding either bound forces eRVS_only fallback (spec §5.2.1.3).
        self.max_analysis_depth = DEFAULT_MAX_ANALYSIS_DEPTH
        self.max_analysis_branches = DEFAULT_MAX_ANALYSIS_BRANCHES
        
        # Artifacts (centralized intermediate files)
        self.artifacts = {
            'walker_metadata': self.artifacts_dir / "walker_metadata.json",
            'walker_template': self.artifacts_dir / "walker_template.cu",
            'llvm_analysis': self.artifacts_dir / "llvm_analysis.json",
            'sync_warnings': self.artifacts_dir / "synchronization_warnings.txt",
            'type_analysis': self.artifacts_dir / "type_analysis.json",
            'llvm_ir': self.artifacts_dir / "walker_template.ll",
            'graph_fields': self.artifacts_dir / "graph_fields.json",
        }
        
        # Generated output files
        self.generated_files = {
            'graph': self.generated_dir / "graph.cuh",
            'gpu_graph': self.generated_dir / "gpu_graph.cuh",
            'fill_dummy': self.generated_dir / "fill_dummy.cuh",
            'get_max_weight': self.generated_dir / "get_max_weight.cuh",
            'get_sum_weight': self.generated_dir / "get_sum_weight.cuh",
            'walker_traits': self.generated_dir / "walker_traits.cuh",
        }

        # Configuration files
        self.graph_fields_config = self.config_dir / "graph_fields.config"
        
        # Tool paths
        self.tools = {
            'metadata_extractor': self.tools_dir / "metadata_extractor",
            'llvm_pass': self.tools_dir / "llvm_pass",
            'type_extractor': self.tools_dir / "type_extractor",
        }
        
        # Build directories (will be cleaned)
        self.build_dirs = ["build", "output"]
        
        # Compilation flags
        self.compile_flags = [
            f"-I{self.include_dir}",
            f"--cuda-gpu-arch={self.cuda_arch}",
            "-DLLVM_ANALYSIS",
            "-std=c++17"
        ]
    
    def ensure_artifacts_dir(self):
        """Ensure artifacts directory exists."""
        self.artifacts_dir.mkdir(exist_ok=True)
    
    def ensure_generated_dir(self):
        """Ensure generated directory exists."""
        self.generated_dir.mkdir(parents=True, exist_ok=True)
    
    def get_artifact_path(self, name: str) -> Path:
        """Get path to a specific artifact."""
        if name not in self.artifacts:
            raise ValueError(f"Unknown artifact: {name}")
        return self.artifacts[name]

    def get_generated_path(self, name: str) -> Path:
        """Get path to a specific generated file."""
        if name not in self.generated_files:
            raise ValueError(f"Unknown generated file: {name}")
        return self.generated_files[name]

    # Convenience properties for common paths
    @property
    def walker_metadata_output(self):
        return self.artifacts['walker_metadata']

    @property
    def walker_template_output(self):
        return self.artifacts['walker_template']

    @property
    def llvm_analysis_output(self):
        return self.artifacts['llvm_analysis']

    @property
    def type_analysis_output(self):
        return self.artifacts['type_analysis']

    @property
    def graph_fields_output(self):
        return self.artifacts['graph_fields']
