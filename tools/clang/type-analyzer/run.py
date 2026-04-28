import os
import subprocess
import shutil

base_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, "../.."))

def ensure_dir(DIR):
    if os.path.exists(DIR):
        shutil.rmtree(DIR)
    os.makedirs(DIR)

def build_ast_tool():
    build_dir = os.path.join(base_dir, "build")
    ensure_dir(build_dir)

    print("[RUN] Running cmake...")
    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)

    print("[RUN] Running make...")
    subprocess.run(["make", "-j"], cwd=build_dir, check=True)
    print("[OK] AST tool built.")

def run_ast_tool():
    output_dir = os.path.join(base_dir, "output")
    ensure_dir(output_dir)

    executable = os.path.join(base_dir, "build", "ast-tool")
    dummy_entry = os.path.join(root_dir, "codegen/pre_codegen/output", "walker_dummy.cu")
    config_path = os.path.join(root_dir, "llvm-pass/output", "walker_analysis.json")  # or update this
    json_output = os.path.join(output_dir, "walker_return.json")

    print("[RUN] Running AST analysis tool...")
    subprocess.run([
        executable,
        dummy_entry,
        "-config", config_path,
        "-o", json_output,
        "--",
        f"-I{os.path.join(root_dir, 'include')}",
        "-DLLVM_ANALYSIS",
        "--cuda-gpu-arch=sm_86"
    ], check=True)
    print(f"[OK] Analysis written to {json_output}")

if __name__ == "__main__":
    build_ast_tool()
    run_ast_tool()
