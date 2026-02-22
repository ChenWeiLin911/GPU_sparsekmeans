import graphblas as gb
from graphblas import Matrix, Vector, dtypes
import scipy.sparse as sp
import time 
import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np
from pathlib import Path
import os
import argparse
import random
import matplotlib.pyplot as plt
import gc

def row_l2_normalize_csr(X: sp.csr_matrix):
    # row-wise L2 norm
    row_norm = np.sqrt(X.multiply(X).sum(axis=1)).A1  # shape (n_rows,)
    

    # broadcasting divide
    X_norm = X.multiply(1.0 / row_norm[:, None]).tocsr()
    return X_norm

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def topk_relative_error(
    A, B,
    k=5,
    eps=1e-15,
    bins=100,
    log_hist=True,
    ignore_below=0.0,
    save_path="rel_error_hist.png"
):


    #A = np.asarray(A)
    #B = np.asarray(B)
    assert A.shape == B.shape

    abs_diff = np.abs(A - B)
    rel_err = abs_diff / np.maximum(np.abs(B), eps)

    flat = rel_err.ravel()
    idxs = np.argpartition(flat, -k)[-k:]
    idxs = idxs[np.argsort(flat[idxs])[::-1]]

    print("Top-k relative errors:")
    for rank, fi in enumerate(idxs, 1):
        idx = np.unravel_index(fi, rel_err.shape)
        print(
            f"[{rank}] idx={idx} "
            f"A={A[idx]} B={B[idx]} "
            f"abs_diff={abs_diff[idx]} "
            f"rel_err={rel_err[idx]:e}"
        )


    rel_flat = rel_err.ravel()

    if ignore_below > 0:
        rel_flat = rel_flat[rel_flat >= ignore_below]

    rel_flat = np.maximum(rel_flat, eps)  # 避免 log10(0)


    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))

    if log_hist:
        plt.hist(np.log10(rel_flat), bins=bins)
        plt.xlabel("log10(relative error)")
        plt.title("Relative error distribution (log scale)")
    else:
        plt.hist(rel_flat, bins=bins)
        plt.xlabel("relative error")
        plt.title("Relative error distribution")

    plt.ylabel("count")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Histogram saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--n_clusters", type=int)
    args = parser.parse_args()
    dataset = args.dataset
    K = args.n_clusters

    end = cp.cuda.Event()

    result_dir = Path(f"./result/{dataset}")
    result_dir.mkdir(parents=True, exist_ok=True)

    X = sp.load_npz(f"../smalldataset/{dataset}_tfidf_train_X_csr.npz")
    gX = gb.io.from_scipy_sparse(X)
    gpu_X = csp.csr_matrix(X)
    B = sp.load_npz(f"../smalldataset/{dataset}-B-iter5-K{K}.npz")
    gB = gb.io.from_scipy_sparse(B.astype(np.float64))
    B_dense = B.toarray()
    gpu_B = cp.asarray(B_dense)

    del B 
    del B_dense 
    del X
    gc.collect()

    D = gpu_B @ gpu_X
    D_np = cp.asnumpy(D)
    end.record()
    end.synchronize()

    BX = Matrix(dtypes.FP64, nrows = gB.nrows, ncols=gX.ncols)

    # GraphBlas normal
    BX << 0
    gB = gB.ss.export("csc")
    gB = gb.Matrix.ss.import_csc(**gB)
    BX(nthreads=8) << gB.mxm(gX)
    BX_np = BX.to_dense(fill_value=0)
    print("finish XCT")
    topk_relative_error(
    BX_np,
    D_np,
    k=5,
    bins=120,
    log_hist=True,
    ignore_below=1e-17,
    save_path=f"./result/{dataset}/rel_error_hist_BX_vs_CuPy_normal_{K}.png"
    )
    
    del BX_np
    gc.collect()

    # GraphBlas accum
    BX << 0
    gB(mask=~gB.S) << 0
    gB = gB.ss.export("fullc")
    gB = gb.Matrix.ss.import_fullc(**gB)
    BX(accum=gb.binary.plus, nthreads=8) << gB.mxm(gX)
    BX_np = BX.to_dense(fill_value=0)
    topk_relative_error(
    BX_np,
    D_np,
    k=5,
    bins=120,
    log_hist=True,
    ignore_below=1e-20,
    save_path=f"./result/{dataset}/rel_error_hist_BX_vs_CuPy_accum_{K}.png"
    )



if __name__ == "__main__":
    main()