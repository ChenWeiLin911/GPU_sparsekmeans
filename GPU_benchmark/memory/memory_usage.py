import gc
import graphblas as gb
import scipy.sparse as sp
import time
import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np
import argparse
from pathlib import Path
import os
import random
import time


"""
Note !!!
Incur data type transfer (ex. fp32 -> fp64) may let calculate not precise.
"""

def csr_matrix_size(X: csp.csr_matrix):
    return X.data.nbytes + X.indices.nbytes + X.indptr.nbytes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--n_clusters", type=int)
    args = parser.parse_args()
    dataset = args.dataset
    K = args.n_clusters

    result_dir = Path(f"./result/{dataset}/{K}")
    result_dir.mkdir(parents=True, exist_ok=True)

    file = open(result_dir / f"memoryusage.txt", "w")
    X = sp.load_npz(f"../smalldataset/{dataset}_tfidf_train_X_csr.npz")
    before = cp.get_default_memory_pool().used_bytes()
    gpu_X = csp.csr_matrix(X)
    after = cp.get_default_memory_pool().used_bytes()
    file.write(f"allocate X: {(after - before) / (1024 ** 3)} GB\n")
    file.write(f"X cost GPU memory: {csr_matrix_size(gpu_X)/(1024*1024*1024)} GB\n")
    #print(f"X cost GPU memory: {csr_matrix_size(gpu_X)/(1024*1024*1024)} GB")
    #print(cp.get_default_memory_pool().used_bytes() / 1024**3, "GiB used")
    #print(cp.get_default_memory_pool().total_bytes() / 1024**3, "GiB allocated")
    C = sp.load_npz(f"../smalldataset/{dataset}-C-iter5-K{K}.npz")
    C_dense = C.toarray()
    before = cp.get_default_memory_pool().used_bytes()
    gpu_C = cp.asarray(C_dense)
    after = cp.get_default_memory_pool().used_bytes()
    file.write(f"allocate C: {(after - before)/ (1024**2)} MB\n")
    before = cp.get_default_memory_pool().total_bytes()
    D = gpu_X @ gpu_C.T
    after = cp.get_default_memory_pool().total_bytes()
    file.write(f"SpMM cost: {(after - before) / (1024 ** 2)} MB\n")
    file.write(f"D cost: {D.nbytes / (1024 ** 2)} MB\n")
    file.write(f"Buffer cost:{(after - before - D.nbytes) / (1024 ** 2)} MB")
    file.close()
if __name__ == "__main__":
    main()