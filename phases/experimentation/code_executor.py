import subprocess
import os
from pathlib import Path
from typing import List, Optional
from phases.experimentation.experiment_state import ExecutionResult


class CodeExecutor:
    """Execution wrapper for running Python code in subprocess."""
    
    def __init__(self, default_timeout: int = 300):
        self.default_timeout = default_timeout
    
    def execute_file(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """Execute Python file in subprocess and return results."""
        
        if timeout is None:
            timeout = self.default_timeout
        
        # Use output_dir if specified, otherwise use directory of the file
        if output_dir is None:
            output_dir = os.path.dirname(file_path)
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
        
        # Track files before execution
        plot_files_before = self._list_plot_files(output_dir) if output_dir else []
        result_files_before = self._list_result_files(output_dir) if output_dir else []
        
        try:
            process = subprocess.Popen(
                ["python", file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=output_dir if output_dir else None
            )
            
            stdout, stderr = process.communicate(timeout=timeout)
            return_code = process.returncode
            
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return_code = 1
            stderr = f"Execution timed out after {timeout} seconds.\n{stderr}"
        
        except Exception as e:
            return_code = 1
            stderr = f"Execution error: {str(e)}\n{stderr if 'stderr' in locals() else ''}"
            stdout = stdout if 'stdout' in locals() else ""
        
        plot_files = []
        result_files = []
        
        # Detect generated files
        if output_dir:
            plot_files_after = self._list_plot_files(output_dir)
            result_files_after = self._list_result_files(output_dir)
            
            plot_files = [f for f in plot_files_after if f not in plot_files_before]
            result_files = [f for f in result_files_after if f not in result_files_before]
        
        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
            plot_files=plot_files,
            result_files=result_files
        )
    
    def _list_plot_files(self, output_dir: str) -> List[str]:
        """List all plot files (PNG, SVG, PDF) in output directory."""
        plot_extensions = {'.png', '.svg', '.pdf', '.jpg', '.jpeg'}
        plot_files = []
        
        plots_dir = os.path.join(output_dir, "plots")
        if os.path.exists(plots_dir):
            for file_path in Path(plots_dir).rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in plot_extensions:
                    plot_files.append(str(file_path))
        
        # Check root output_dir for plots
        if os.path.exists(output_dir):
            for file_path in Path(output_dir).glob('*'):
                if file_path.is_file() and file_path.suffix.lower() in plot_extensions:
                    plot_files.append(str(file_path))
        
        return sorted(plot_files)
    
    def _list_result_files(self, output_dir: str) -> List[str]:
        """List all result files (JSON, CSV) in output directory."""
        result_extensions = {'.json', '.csv'}
        result_files = []
        
        if os.path.exists(output_dir):
            for file_path in Path(output_dir).rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in result_extensions:
                    # Exclude metadata.json
                    if file_path.name != "metadata.json":
                        result_files.append(str(file_path))
        
        return sorted(result_files)

