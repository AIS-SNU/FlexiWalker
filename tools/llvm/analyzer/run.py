import os
import subprocess
import shutil

def ensure_dirs(DIR):
    if os.path.exists(DIR):
        shutil.rmtree(DIR)
    os.makedirs(DIR)

def run_cmake_and_make(llvm_pass_dir):
    build_dir = os.path.join(llvm_pass_dir, "build")

    # Clean and recreate build directory
    ensure_dirs(build_dir)

    print("[RUN] Running CMake for LLVM pass...")
    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)

    print("[RUN] Building LLVM pass (make)...")
    subprocess.run(["make", "-j"], cwd=build_dir, check=True)
    print("[OK] LLVM pass built.")

def run_llvm_codegen():
    # Paths and settings
    LLVM_VERSION = "17"
    cuda_arch = "sm_86"
    base_dir = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    output_dir = os.path.join(base_dir, "output")
    llvm_pass_dir = os.path.join(root_dir, "llvm-pass")

    clang = f"clang++-{LLVM_VERSION}"
    opt = f"opt-{LLVM_VERSION}"

    cuda_path = "/usr/local/cuda"
    include_dir = os.path.join(root_dir, "include")
    llvm_pass_plugin = os.path.join(llvm_pass_dir, "build/AdjwgtDetector.so")

    # Input/output files
    generated_analysis_file = os.path.join(root_dir, "codegen/pre_codegen/output", "walker_dummy.cu")
    ir_output = os.path.join(output_dir, "walk.ll")
    json_output = os.path.join(output_dir, "walker_analysis.json")

    ensure_dirs(output_dir)

    # Step 0: Compile LLVM pass
    run_cmake_and_make(llvm_pass_dir)

    # Step 1: Compile CUDA to LLVM IR
    print("[RUN] Generating LLVM IR with clang...")
    clang_cmd = [
        clang,
        "-g", "-x", "cuda", "-std=c++17",
        "-DLLVM_ANALYSIS", "-O0",
        f"-I{include_dir}",
        f"--cuda-gpu-arch={cuda_arch}",
        "-S", "-emit-llvm",
        "--cuda-device-only", "-nocudalib",
        generated_analysis_file,
        "-o", ir_output
    ]
    subprocess.run(clang_cmd, check=True)
    print(f"[OK] LLVM IR written to {ir_output}")

    # Step 2: Run opt with LLVM pass
    print("[RUN] Running LLVM pass...")
    opt_cmd = [
        opt,
        "-load-pass-plugin", llvm_pass_plugin,
        "-passes=adjwgt-detector",
        ir_output,
        f"-walker-json={json_output}",
        "-disable-output"
    ]
    subprocess.run(opt_cmd, check=True)
    print(f"[OK] JSON emitted to {json_output}" if os.path.exists(json_output) else "[FAIL] JSON file not created")

if __name__ == "__main__":
    run_llvm_codegen()
