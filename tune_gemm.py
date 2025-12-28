
import os
import re
import subprocess

original_file = "cuda/48-gemm-17.cu"
backup_file = "cuda/48-gemm-17.cu.bak"

# Ensure backup
if not os.path.exists(backup_file):
    os.system(f"cp {original_file} {backup_file}")

def run_test(config):
    # Read content
    with open(backup_file, 'r') as f:
        content = f.read()

    # config: {BM, BN, BK, WM, WN, WNITER, TM, TN, THREAD_NUMS}
    
    # Update constants
    content = re.sub(r'constexpr static int BLOCK_M\s*=\s*\d+;', f'constexpr static int BLOCK_M = {config["BM"]};', content)
    content = re.sub(r'constexpr static int BLOCK_N\s*=\s*\d+;', f'constexpr static int BLOCK_N = {config["BN"]};', content)
    content = re.sub(r'constexpr static int BLOCK_K\s*=\s*\d+;', f'constexpr static int BLOCK_K = {config["BK"]};', content)
    content = re.sub(r'constexpr static int THREAD_NUMS\s*=\s*\d+;', f'constexpr static int THREAD_NUMS = {config["THREAD_NUMS"]};', content)

    # Update kernel call
    # Pattern: matrix_multiplication_kernel<BLOCK_M, BLOCK_N, BLOCK_K, 64, 32, 1, 8, 8, THREAD_NUMS>
    # We need to replace the template args: 64, 32, 1, 8, 8
    # which correspond to WM, WN, WNITER, TM, TN
    
    search_pattern = r'matrix_multiplication_kernel<BLOCK_M, BLOCK_N, BLOCK_K, \d+, \d+, \d+, \d+, \d+, THREAD_NUMS>'
    replacement = f'matrix_multiplication_kernel<BLOCK_M, BLOCK_N, BLOCK_K, {config["WM"]}, {config["WN"]}, {config["WNITER"]}, {config["TM"]}, {config["TN"]}, THREAD_NUMS>'
    content = re.sub(search_pattern, replacement, content)

    with open(original_file, 'w') as f:
        f.write(content)

    # Compile
    # -Xptxas -v to see memory usage if needed, but output parsing might be noisy
    # Using -O3 and arch
    cmd = "nvcc -o gemm_test cuda/48-gemm-17.cu -arch=sm_86 -O3"
    ret = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ret.returncode != 0:
        return f"Compile failed: {ret.stderr.decode()}"

    # Run
    cmd = "./gemm_test"
    try:
        ret = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    except subprocess.TimeoutExpired:
        return "Timeout"
    
    if ret.returncode != 0:
        return f"Run failed: {ret.stderr.decode()}"
    
    output = ret.stdout.decode()
    # Extract TFLOPS
    m = re.search(r'Compute Performance: ([\d.]+) TFLOPS', output)
    if m:
        return float(m.group(1))
    else:
        return f"Parse failed. Output: {output}"

configs = [
    # Baseline
    {"BM": 128, "BN": 128, "BK": 16, "WM": 64, "WN": 32, "WNITER": 1, "TM": 8, "TN": 8, "THREAD_NUMS": 256},
    # Candidate 1: Increase BM
    {"BM": 256, "BN": 128, "BK": 16, "WM": 64, "WN": 64, "WNITER": 1, "TM": 8, "TN": 8, "THREAD_NUMS": 256},
    # Candidate 2: Increase BN
    {"BM": 128, "BN": 256, "BK": 16, "WM": 64, "WN": 64, "WNITER": 1, "TM": 8, "TN": 8, "THREAD_NUMS": 256},
    # Candidate 3: BK=32 (Small Block to fit SMEM)
    {"BM": 64, "BN": 64, "BK": 32, "WM": 32, "WN": 32, "WNITER": 1, "TM": 8, "TN": 4, "THREAD_NUMS": 128}, # 64x64 / 32x32 = 2x2 = 4 warps = 128 threads
    # Candidate 4: Larger threads?
    {"BM": 128, "BN": 128, "BK": 16, "WM": 64, "WN": 64, "WNITER": 1, "TM": 8, "TN": 8, "THREAD_NUMS": 128}, # 4 warps
    # Candidate 5: WNITER=2 ?
    # If WNITER=2, WSUBN changes.
    
    # Candidate 6: BK=32, larger BM
    {"BM": 128, "BN": 64, "BK": 32, "WM": 64, "WN": 32, "WNITER": 1, "TM": 8, "TN": 4, "THREAD_NUMS": 128},
    
    # Candidate 7: Standard BM/BN with slightly different Warp configuration
    {"BM": 128, "BN": 128, "BK": 8, "WM": 64, "WN": 32, "WNITER": 1, "TM": 8, "TN": 8, "THREAD_NUMS": 256},

    # Candidate 8: Maximize threads per block (smaller work per thread)
    # BM=128, BN=128. WM=32, WN=32. 4x4 warps = 16 warps = 512 threads.
    # LHS_WMITER = (32*32)/(32*8*4) = 1 (if TM=8, TN=4)
    {"BM": 128, "BN": 128, "BK": 16, "WM": 32, "WN": 32, "WNITER": 1, "TM": 8, "TN": 4, "THREAD_NUMS": 512},

    # Candidate 9: BM=256, BK=8
    {"BM": 256, "BN": 128, "BK": 8, "WM": 64, "WN": 64, "WNITER": 1, "TM": 8, "TN": 8, "THREAD_NUMS": 256},

    # Candidate 10: BM=128, BN=256, BK=8
    {"BM": 128, "BN": 256, "BK": 8, "WM": 64, "WN": 64, "WNITER": 1, "TM": 8, "TN": 8, "THREAD_NUMS": 256},
    
    # Candidate 14: TM=16, TN=8. WM=128. THREADS=128.
    {"BM": 128, "BN": 128, "BK": 16, "WM": 128, "WN": 32, "WNITER": 1, "TM": 16, "TN": 8, "THREAD_NUMS": 128},

    # Candidate 15: TM=8, TN=16. WM=64, WN=64. THREADS=128.
    {"BM": 128, "BN": 128, "BK": 16, "WM": 64, "WN": 64, "WNITER": 1, "TM": 8, "TN": 16, "THREAD_NUMS": 128},

    # Candidate 16: TM=8, TN=16. BM=128, BN=256. THREADS=256.
    {"BM": 128, "BN": 256, "BK": 16, "WM": 64, "WN": 64, "WNITER": 1, "TM": 8, "TN": 16, "THREAD_NUMS": 256},
    
    # Candidate 17: Same as 16 but BK=8
    {"BM": 128, "BN": 256, "BK": 8, "WM": 64, "WN": 64, "WNITER": 1, "TM": 8, "TN": 16, "THREAD_NUMS": 256},
]

results = []
for c in configs:
    print(f"Testing config: {c}")
    res = run_test(c)
    print(f"Result: {res}")
    results.append((c, res))

best = max(results, key=lambda x: x[1] if isinstance(x[1], float) else -1)
print(f"Best: {best}")

# Restore original if needed, or leave best?
# For now, let's just find the best.

