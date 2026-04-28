"""FlexiWalker compilation pipeline package."""

from .config import PipelineConfig
from .base import PipelineStage, PipelineError, ToolBuildManager, setup_logging
from .stages import (
    DummyGenerator,
    MetadataExtractor,
    TemplateGenerator,
    LLVMAnalyzer,
    TypeExtractor,
    GraphFieldDetectorStage,
    CodeGenerator
)

__all__ = [
    'PipelineConfig',
    'PipelineStage',
    'PipelineError',
    'ToolBuildManager',
    'setup_logging',
    'DummyGenerator',
    'MetadataExtractor',
    'TemplateGenerator',
    'LLVMAnalyzer',
    'TypeExtractor',
    'GraphFieldDetectorStage',
    'CodeGenerator'
]
