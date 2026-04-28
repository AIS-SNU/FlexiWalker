#!/usr/bin/env python3
"""
FlexiWalker Compilation Pipeline

A restructured, maintainable pipeline for the FlexiWalker compiler.
This replaces run_full_pipeline.py with proper dependency management,
centralized artifacts, and better error handling.
"""

import argparse
import shutil
import sys
from pathlib import Path

from pipeline import (
    PipelineConfig,
    PipelineError,
    setup_logging,
    DummyGenerator,
    MetadataExtractor,
    TemplateGenerator,
    LLVMAnalyzer,
    TypeExtractor,
    GraphFieldDetectorStage,
    CodeGenerator
)


class PipelineRunner:
    """Main pipeline runner with proper dependency management."""
    
    def __init__(self, root_dir: str, config: PipelineConfig):
        self.root_dir = Path(root_dir)
        self.config = config
        
        # Define pipeline stages in dependency order
        self.stages = [
            DummyGenerator(config),            # MUST run first - generates dummy files needed by all other stages
            MetadataExtractor(config),
            TemplateGenerator(config),
            LLVMAnalyzer(config),
            TypeExtractor(config),
            GraphFieldDetectorStage(config),  # Detect fields from graph_fields.config
            CodeGenerator(config)
        ]
    
    def clean_build_directories(self):
        """Clean up build directories across the project."""
        print("[CLEAN] Cleaning build directories...")

        cleaned_dirs = []

        # Clean directories matching build_dirs pattern
        for dirpath in self.root_dir.rglob("*"):
            if dirpath.is_dir() and dirpath.name in self.config.build_dirs:
                print(f"  [RM] Removing {dirpath}")
                shutil.rmtree(dirpath, ignore_errors=True)
                cleaned_dirs.append(str(dirpath))

        # Clean artifacts directory
        if self.config.artifacts_dir.exists():
            print(f"  [RM] Removing {self.config.artifacts_dir}")
            shutil.rmtree(self.config.artifacts_dir, ignore_errors=True)
            cleaned_dirs.append(str(self.config.artifacts_dir))

        # Clean generated directory
        if self.config.generated_dir.exists():
            print(f"  [RM] Removing {self.config.generated_dir}")
            shutil.rmtree(self.config.generated_dir, ignore_errors=True)
            cleaned_dirs.append(str(self.config.generated_dir))

        if cleaned_dirs:
            print(f"[OK] Cleaned {len(cleaned_dirs)} directories")
        else:
            print("[OK] No directories to clean")
    
    def run_stage(self, stage, force: bool = False):
        """Run a single stage with error handling."""
        try:
            stage.run(force=force)
        except PipelineError as e:
            print(f"[FAIL] Stage '{stage.name}' failed: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Unexpected error in stage '{stage.name}': {e}")
            sys.exit(1)
    
    def run_pipeline(self, force: bool = False, stage_filter: str = None):
        """Run the complete pipeline or specific stages."""
        print("[START] Starting FlexiWalker compilation pipeline")
        print(f"  Root directory:      {self.root_dir}")
        print(f"  Artifacts directory: {self.config.artifacts_dir}")
        print(f"  Generated directory: {self.config.generated_dir}")
        print()

        # Filter stages if requested
        stages_to_run = self.stages
        if stage_filter:
            stages_to_run = [s for s in self.stages if stage_filter in s.name]
            if not stages_to_run:
                print(f"[FAIL] No stages match filter: {stage_filter}")
                sys.exit(1)
            print(f"[FILTER] Running filtered stages: {[s.name for s in stages_to_run]}")

        # Run stages
        for i, stage in enumerate(stages_to_run, 1):
            print(f"\n[{i}/{len(stages_to_run)}] Running stage: {stage.name}")
            self.run_stage(stage, force=force)

        print(f"\n[OK] Pipeline completed successfully")
        print(f"Generated files:")
        for name, path in self.config.generated_files.items():
            if path.exists():
                print(f"  [OK]      {name}: {path}")
            else:
                print(f"  [MISSING] {name}: {path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FlexiWalker Compilation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run pipeline (skips up-to-date stages and tools)
  %(prog)s --clean            # Delete everything and rebuild from scratch
  %(prog)s --force            # Re-run all stages (tools still cached if unchanged)
  %(prog)s --stage metadata   # Run only metadata-related stages
  %(prog)s --verbose          # Enable verbose logging output
  %(prog)s --clean --force    # Nuclear option: clean + force all stages
        """
    )
    
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Delete all artifacts, generated files, and tool builds before running (full clean rebuild)"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force all pipeline stages to run (ignore stage output timestamps, but tools still use smart caching)"
    )
    
    parser.add_argument(
        "--stage", "-s",
        type=str,
        help="Filter stages by name (substring match)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List all available stages and exit"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    # Get root directory
    root_dir = Path(__file__).parent.resolve()
    
    # Create configuration
    config = PipelineConfig(root_dir)
    
    # Create pipeline runner
    runner = PipelineRunner(root_dir, config)
    
    # List stages if requested
    if args.list_stages:
        print("Available pipeline stages:")
        for i, stage in enumerate(runner.stages, 1):
            print(f"  {i}. {stage.name}")
            print(f"     Dependencies: {[str(d) for d in stage.dependencies]}")
            print(f"     Outputs: {[str(o) for o in stage.outputs]}")
            print()
        return
    
    try:
        # Clean if requested
        if args.clean:
            runner.clean_build_directories()
        
        # Run pipeline
        runner.run_pipeline(force=args.force, stage_filter=args.stage)
        
    except KeyboardInterrupt:
        print("\n[ABORT] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
