"""LLVM analysis stage - analyzes LLVM IR for optimization information."""

import json
from pathlib import Path
from typing import List, Dict, Any

from ..base import PipelineStage, PipelineError, ToolBuildHelper


class LLVMAnalyzer(PipelineStage):
    """Analyze LLVM IR to extract optimization information.
    
    This stage compiles walker template code to LLVM IR and runs
    a custom LLVM pass to extract analysis data for optimization.
    """
    
    @property
    def name(self) -> str:
        return "llvm_analysis"
    
    @property
    def dependencies(self) -> List[Path]:
        return [self.config.get_artifact_path('walker_template')]
    
    @property
    def outputs(self) -> List[Path]:
        return [
            self.config.get_artifact_path('llvm_analysis'),
            self.config.get_artifact_path('llvm_ir')
        ]
    
    def _tool(self) -> ToolBuildHelper:
        return ToolBuildHelper(
            stage=self,
            tool_dir=self.config.root_dir / "tools" / "llvm" / "analyzer",
            executable_name="AdjwgtDetector.so",
        )

    def _get_clang_cmd(self) -> str:
        """Get the clang command name."""
        return f"clang++-{self.config.llvm_version}"
    
    def _get_opt_cmd(self) -> str:
        """Get the opt command name."""
        return f"opt-{self.config.llvm_version}"
    
    def _compile_to_llvm_ir(self):
        """Compile CUDA code to LLVM IR.
        
        Compiles the walker template code to LLVM IR for analysis.
        Uses clang with CUDA support to generate device-only IR.
        
        Raises:
            PipelineError: If compilation fails or input file is missing
        """
        input_file = self.config.get_artifact_path('walker_template')
        output_file = self.config.get_artifact_path('llvm_ir')
        
        # Validate input file exists
        if not input_file.exists():
            raise PipelineError(f"Walker template not found: {input_file}")
        
        clang = self._get_clang_cmd()
        cmd = [
            clang,
            "-g", "-x", "cuda", "-std=c++17",
            "-DLLVM_ANALYSIS", "-O0",
            f"-I{self.config.include_dir}",
            f"--cuda-gpu-arch={self.config.cuda_arch}",
            "-S", "-emit-llvm",
            "--cuda-device-only", "-nocudalib",
            str(input_file),
            "-o", str(output_file)
        ]
        
        try:
            self.run_command(cmd)
            
            # Verify output was created
            if not output_file.exists():
                raise PipelineError(f"LLVM IR file was not generated: {output_file}")
                
            self.logger.info(f"Generated LLVM IR: {output_file}")
            self.logger.debug(f"IR file size: {output_file.stat().st_size} bytes")
            
        except Exception as e:
            raise PipelineError(f"Failed to compile to LLVM IR: {e}") from e
    
    def _run_llvm_pass(self):
        """Run the LLVM pass to extract analysis information.
        
        Executes the custom LLVM pass on the generated IR to extract
        optimization analysis data.
        
        Raises:
            PipelineError: If pass execution fails or required files are missing
        """
        tool = self._tool()
        ir_file = self.config.get_artifact_path('llvm_ir')
        analysis_file = self.config.get_artifact_path('llvm_analysis')

        # Validate inputs exist
        if not ir_file.exists():
            raise PipelineError(f"LLVM IR file not found: {ir_file}")
        if not tool.executable_path.exists():
            raise PipelineError(f"LLVM pass not found: {tool.executable_path}")

        opt = self._get_opt_cmd()
        sync_warnings_file = self.config.get_artifact_path('sync_warnings')
        cmd = [
            opt,
            "-load-pass-plugin", str(tool.executable_path),
            "-passes=adjwgt-detector",
            str(ir_file),
            f"-walker-json={analysis_file}",
            f"-walker-sync-warnings={sync_warnings_file}",
            "-disable-output"
        ]
        
        try:
            self.run_command(cmd)
            
            # Verify output was created
            if not analysis_file.exists():
                raise PipelineError(f"LLVM analysis file was not generated: {analysis_file}")
                
            self.logger.info(f"LLVM analysis completed: {analysis_file}")
            self.logger.debug(f"Analysis file size: {analysis_file.stat().st_size} bytes")
            
        except Exception as e:
            raise PipelineError(f"LLVM pass execution failed: {e}") from e
    
    def execute(self):
        """Execute LLVM analysis."""
        # Ensure artifacts directory exists
        self.config.ensure_artifacts_dir()
        
        # Build LLVM pass
        self._tool().ensure_built()

        # Compile to LLVM IR
        self._compile_to_llvm_ir()
        
        # Run LLVM pass
        self._run_llvm_pass()
        
        self.logger.info("LLVM analysis completed")
    
    def validate_output_content(self):
        """Validate that the LLVM analysis produced valid output.
        
        Checks that both LLVM IR and analysis files exist and contain
        expected data structures.
        """
        super().validate_output_content()
        
        # Validate LLVM IR file
        ir_file = self.config.get_artifact_path('llvm_ir')
        if ir_file.exists():
            try:
                with open(ir_file, 'r') as f:
                    ir_content = f.read()
                if not ir_content.strip():
                    self.logger.warning("LLVM IR file appears to be empty")
                elif 'define' not in ir_content:
                    self.logger.warning("LLVM IR does not appear to contain function definitions")
                else:
                    self.logger.debug(f"LLVM IR validation passed - {len(ir_content)} characters")
            except Exception as e:
                self.logger.warning(f"Could not validate LLVM IR file: {e}")
        
        # Validate analysis file
        analysis_file = self.config.get_artifact_path('llvm_analysis')
        if analysis_file.exists():
            try:
                with open(analysis_file, 'r') as f:
                    data = json.load(f)
                
                if not isinstance(data, dict):
                    self.logger.warning(f"LLVM analysis should be a dictionary, got {type(data)}")
                elif len(data) == 0:
                    self.logger.warning("LLVM analysis file appears to be empty")
                else:
                    self.logger.debug(f"LLVM analysis validation passed - {len(data)} entries")
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"LLVM analysis file contains invalid JSON: {e}")
            except Exception as e:
                self.logger.warning(f"Could not validate LLVM analysis file: {e}")
