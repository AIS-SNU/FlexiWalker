import os
import shutil
import subprocess

# Configurable paths
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(TOOL_DIR, "../..")
OUTPUT_DIR = os.path.join(TOOL_DIR, "output")
DUMMY_FILE = os.path.join(OUTPUT_DIR, "dummy_entry.cu")
BUILD_DIR = os.path.join(TOOL_DIR, "build")
WALKER_TOOL = os.path.join(BUILD_DIR, "pre-tool")
JSON_OUTPUT = os.path.join(OUTPUT_DIR, "walker_meta.json")

def ensure_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)

def create_dummy_entry():
    with open(DUMMY_FILE, "w") as f:
        f.write('#include "app.cuh"\n')
    print(f"[OK] Created {DUMMY_FILE}")

def build_ast_tool():
    print("[RUN] Running CMake...")
    subprocess.run(["cmake", ".."], cwd=BUILD_DIR, check=True)

    print("[RUN] Building walker_ast...")
    subprocess.run(["make", "-j"], cwd=BUILD_DIR, check=True)

def run_ast_tool():
    print("[RUN] Running walker_ast on dummy_entry.cu...")
    subprocess.run([
        WALKER_TOOL,
        DUMMY_FILE,
        "-o", JSON_OUTPUT,
        "--",
        f"-I{os.path.join(BASE_DIR, 'include')}",
        "--cuda-gpu-arch=sm_86",
        "-DLLVM_ANALYSIS"
    ], check=True)

    if os.path.exists(JSON_OUTPUT):
        print(f"[OK] walker_meta.json written to {JSON_OUTPUT}")
    else:
        print("[FAIL] walker_meta.json not found - did the tool run correctly?")

def main():
    ensure_dirs()
    script_path = os.path.join(TOOL_DIR, "../../codegen/pre_codegen/_run.py")
    subprocess.run(["python3", script_path], check=True)  # Will raise if error
    create_dummy_entry()
    build_ast_tool()
    run_ast_tool()

if __name__ == "__main__":
    main()
