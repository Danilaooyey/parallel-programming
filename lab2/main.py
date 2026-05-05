import numpy as np
import subprocess
import sys
import os
import re

def generateMatrix(filename, N):
    mat = np.random.rand(N, N)
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        np.savetxt(f, mat, fmt='%.6f')
    return mat

def readMatrix(filename):
    with open(filename, 'r') as f:
        N = int(f.readline().strip())
        mat = np.loadtxt(f)
    return mat

def runExperiment(N, threads, exePath, verify=False):
    fileA = "A.txt"
    fileB = "B.txt"
    fileC = "out.txt"

    matA = generateMatrix(fileA, N)
    matB = generateMatrix(fileB, N)

    result = subprocess.run([exePath, fileA, fileB, fileC, str(threads)], capture_output=True, text=True, encoding='utf-8')
    
    timeMatch = re.search(r"Lead time:\s+([0-9.]+)", result.stdout)
    exec_time = float(timeMatch.group(1)) if timeMatch else -1

    if verify:
        matCPython = np.dot(matA, matB)
        matC = readMatrix(fileC)
        is_correct = np.allclose(matCPython, matC, atol=1e-5)
        if not is_correct:
            print(f"VERIFICATION ERROR on N={N}, threads={threads}!")
            sys.exit(1)

    os.remove(fileA)
    os.remove(fileB)
    os.remove(fileC)
    
    return exec_time

if __name__ == "__main__":
    cpp_executable = "./matrix_omp" if os.name != 'nt' else "matrix_omp.exe"
    
    if not os.path.exists(cpp_executable):
        print(f"Error: Executable file '{cpp_executable}' not found.")
        print("Compile: g++ -O3 -fopenmp matrix_omp.cpp -o matrix_omp")
        sys.exit(1)

    sizes_to_test = [200, 400, 800, 1200, 1600, 2000]
    threads_to_test = [1, 2, 4, 6, 8, 12]
    
    results = {n: {} for n in sizes_to_test}

    print("Conducting quick verification for N=400...")
    runExperiment(400, 4, cpp_executable, verify=True)
    print("Verification completed successfully. Let's begin the benchmark...\n")

    header = f"| N \\ Streams | " + " | ".join([str(t) for t in threads_to_test]) + " |"
    print(header)
    print("|---" + "|---" * len(threads_to_test) + "|")

    for size in sizes_to_test:
        row_str = f"| {size} |"
        for threads in threads_to_test:
            exec_time = runExperiment(size, threads, cpp_executable, verify=False)
            results[size][threads] = exec_time
            row_str += f" {exec_time:.5f} |"
        print(row_str)