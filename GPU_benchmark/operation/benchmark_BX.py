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
    B = sp.load_npz(f"../smalldataset/{dataset}-B-iter5-K{K}.npz")

    gX = gb.io.from_scipy_sparse(X)
    gpu_X = csp.csr_matrix(X)
    gB = gb.io.from_scipy_sparse(B)
    BX = Matrix(dtypes.FP64, nrows=gB.nrows, ncols=gX.ncols)
           

    BX_normal = []
    BX_accum = []
    BX_gpu = []


    # GraphBlas normal
    for _ in range(5):
        BX << 0
        start_time = time.time()
        BX(nthreads=8) << gB.mxm(gX)
        end_time = time.time()
        BX_normal.append(end_time - start_time)
        print(f"BX normal cost: {end_time - start_time}")

    
    for _ in range(5):
    # GraphBlas accum
        start_time = time.time()
        BX(accum=gb.binary.plus, nthreads=8) << gB.mxm(gX)
        end_time = time.time()
        BX_accum.append(end_time - start_time)
        print(f"BX accum cost: {end_time - start_time}")


    for _ in range(5):
            
        # ---GPU Test---
        B_dense = B.toarray()
        start_time = time.time()
        end = cp.cuda.Event()
        gpu_B = cp.asarray(B_dense)
        _ = gpu_B @ gpu_X
        end.record()
        end.synchronize()
        end_time = time.time()
        BX_gpu.append(end_time - start_time)
        print(f"BX GPU cost: {end_time - start_time}")
  
    # release resource
    del B
    del B_dense
    del gpu_B
    gB.clear()
    del gB
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

        
    print(f"GraphBlas normal Mean Runtime: {np.array(BX_normal).mean()}")
    print(f"GraphBlas accum Mean Runtime: {np.array(BX_accum).mean()}")
    print(f"GPU Mean Runtime: {np.median(np.array(BX_gpu))}")
        
    with open(result_dir / f"BX_report.txt", 'w') as f:
        f.write(f"GraphBlas normal Mean Runtime: {np.array(BX_normal).mean()}\n")
        f.write(f"GraphBlas accum Mean Runtime: {np.array(BX_accum).mean()}\n")
        f.write(f"GPU median Runtime: {np.median(np.array(BX_gpu))}\n")


if __name__ == "__main__":
    main()