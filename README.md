# GPU_sparsekmeans

This repository focuses on GPU benchmarking and performance analysis of sparse K-means related operations.

It includes:
- `GPU_benchmark`: benchmark and analysis scripts comparing GraphBLAS and CuPy for key sparse/dense operations used in K-means.
- `sparsekmeans`: an existing sparse K-means implementation (Lloyd and Elkan variants), used as a baseline for benchmarking and numerical checks.

## Repository Layout

```text
.
├── install.sh
├── sparsekmeans/
│   ├── setup.py
│   ├── LICENSE
│   ├── README.md
│   └── sparsekmeans/
│       ├── __init__.py
│       └── sparse_kmeans.py
└── GPU_benchmark/
    ├── operation/
    │   ├── benchmark_BX.py
    │   ├── benchmark_BX_SpGEMM.py
    │   ├── benchmark_XCT.py
    │   ├── BX_run.sh
    │   ├── BX_SpGEMM_run.sh
    │   └── XCT_run.sh
    ├── memory/
    │   ├── memory_usage.py
    │   └── mem_run.sh
    ├── numericaldiff/
    │   ├── nm.py
    │   └── nm_run.sh
    └── smalldataset/   # downloaded by install.sh
```

## Requirements

- Python >= 3.10
- Conda (recommended)
- NVIDIA GPU with a working CUDA driver (`nvidia-smi`)

`sparsekmeans` dependencies:
- `numpy`
- `scipy`
- `python-graphblas`

Benchmark dependencies:
- `cupy`
- `matplotlib`
- `gdown` (for dataset download in `install.sh`)

## Installation

From the repository root:

```bash
bash install.sh
```

This script will:
- create conda env `gpu_sparsekmeans` (Python 3.10),
- install `sparsekmeans` in editable mode,
- install CuPy from conda-forge,
- install `matplotlib` and `gdown`,
- download and unzip benchmark data into `GPU_benchmark/smalldataset/`.

Activate the environment:

```bash
conda activate gpu_sparsekmeans
```

Note: CuPy must match your CUDA/driver and GPU architecture. If you hit `CUDA_ERROR_NO_BINARY_FOR_GPU`, reinstall CuPy with a newer CUDA build.

## Reference Implementation: `sparsekmeans`

The `sparsekmeans` package is included as a baseline implementation for comparisons.

```python
from sparsekmeans import LloydKmeans, ElkanKmeans, kmeans_predict

# X can be scipy.sparse.csr_matrix or graphblas.Matrix
kmeans = LloydKmeans(
    n_clusters=100,
    n_threads=8,
    max_iter=300,
    tol=1e-4,
    random_state=0,
)

labels = kmeans.fit(X)
test_labels = kmeans.predict(X_test)

# Or with user-defined centroids
pred = kmeans_predict(X_test, centroids)
```

Main API:
- `LloydKmeans`
- `ElkanKmeans`
- `kmeans_predict(X, centroids, n_threads=...)`

## GPU Benchmark Scripts

The benchmarks focus on:
- operator-level performance (GraphBLAS vs CuPy),
- sparse vs dense computation trade-offs,
- memory usage,
- numerical consistency across backends.

Output locations:

```text
GPU_benchmark/operation/result/<dataset>/<n_clusters>/
GPU_benchmark/memory/result/<dataset>/<n_clusters>/
GPU_benchmark/numericaldiff/result/<dataset>/
```

## Expected Input Files

Benchmark scripts load from `../smalldataset/` relative to each benchmark folder.
Expected files:
- `<dataset>_tfidf_train_X_csr.npz`
- `<dataset>-B-iter5-K<K>.npz`
- `<dataset>-C-iter5-K<K>.npz`

`install.sh` downloads these automatically into `GPU_benchmark/smalldataset/`.

## Operation Benchmarks

Run from `GPU_benchmark/operation`:

```bash
python3 -u benchmark_BX.py --dataset eurlex --n_clusters 100
python3 -u benchmark_BX_SpGEMM.py --dataset eurlex --n_clusters 100
python3 -u benchmark_XCT.py --dataset eurlex --n_clusters 100
```

Batch scripts:
- `BX_run.sh`
- `BX_SpGEMM_run.sh`
- `XCT_run.sh`

## Memory Benchmark

Run from `GPU_benchmark/memory`:

```bash
python3 -u memory_usage.py --dataset eurlex --n_clusters 100
```

Batch script:
- `mem_run.sh`

## Numerical Difference Check

Run from `GPU_benchmark/numericaldiff`:

```bash
python3 -u nm.py --dataset eurlex --n_clusters 100
```

This compares GraphBLAS and CuPy results and saves relative-error histograms.

Batch script:
- `nm_run.sh`
