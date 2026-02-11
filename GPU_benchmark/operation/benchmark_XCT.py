import gc
import graphblas as gb
from graphblas import Matrix, Vector, dtypes
import scipy.sparse as sp
import time
import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np
import argparse
from pathlib import Path
import os

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--n_clusters", type=int)
    args = parser.parse_args()
    dataset = args.dataset
    K = args.n_clusters

    result_dir = Path(f"./result/{dataset}/{K}")
    result_dir.mkdir(parents=True, exist_ok=True)


    X = sp.load_npz(f"../smalldataset/{dataset}_tfidf_train_X_csr.npz")
    C = sp.load_npz(f"../smalldataset/{dataset}-C-iter5-K{K}.npz")


    gX = gb.io.from_scipy_sparse(X)
    gpu_X = csp.csr_matrix(X)
    gC = gb.io.from_scipy_sparse(C)
    XCT = Matrix(dtypes.FP64, nrows=gX.nrows, ncols=gC.nrows)

    rows = X.shape[0]
    cols = X.shape[1]


    print("X shape:", X.shape)
    print("C shape:", C.shape)

    print("#nnz X:", X.count_nonzero())
    print("#nnz C:", C.count_nonzero())

    print("Sparsity of X: ", X.count_nonzero() / (X.shape[0] * X.shape[1]))
    print("Sparsity of C: ", C.count_nonzero() / (C.shape[0] * C.shape[1]))

    XCT_normal = []
    XCT_accum = []
    XCT_gpu = []

    XCT = Matrix(dtypes.FP64, nrows=gX.nrows, ncols=gC.nrows)

    # GraphBlas normal
    for _ in range(5):
        gC = gC.ss.export("csc")
        gC = gb.Matrix.ss.import_csc(**gC)
        start_time = time.time()
        XCT(nthreads=8) << gX.mxm(gC.T)
        end_time = time.time()
        XCT_normal.append(end_time - start_time)
        print(f"XCT normal cost: {end_time - start_time}")
        

    # GraphBlas accum
    for _ in range(5):
        XCT << 0
        gC(mask=~gC.S) << 0
        gC = gC.ss.export("fullc")
        gC = gb.Matrix.ss.import_fullc(**gC)
        start_time = time.time()
        XCT(accum=gb.binary.plus, nthreads=8) << gX.mxm(gC.T)
        end_time = time.time()
        XCT_accum.append(end_time - start_time)
        print(f"XCT accum cost: {end_time - start_time}")


    # ---GPU Test---
    for _ in range(5):
        C_dense = C.toarray()
        start_time = time.time()
        end = cp.cuda.Event()
        gpu_C = cp.asarray(C_dense)
        _ = gpu_X @ gpu_C.T
        end.record()
        end.synchronize()
        end_time = time.time()
        XCT_gpu.append(end_time - start_time)
        print(f"XCT GPU cost: {end_time - start_time}")



    # release resource
    del C
    del C_dense
    del gpu_C
    gC.clear()
    del gC
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


    print(f"GraphBlas normal Mean Runtime: {np.array(XCT_normal).mean()}")
    print(f"GraphBlas accum Mean Runtime: {np.array(XCT_accum).mean()}")
    print(f"GPU Median Runtime: {np.median(np.array(XCT_gpu))}")


    with open(result_dir / "XCT_report.txt", 'w') as f:
        f.write(f"GraphBlas normal Mean Runtime: {np.array(XCT_normal).mean()}\n")
        f.write(f"GraphBlas accum Mean Runtime: {np.array(XCT_accum).mean()}\n")
        f.write(f"GPU Median Runtime: {np.median(np.array(XCT_gpu))}\n")


if __name__ == "__main__":
    main()
