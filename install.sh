#!/usr/bin/env bash
conda create -n gpu_sparsekmeans python=3.10 -y

cd sparsekmeans
conda run -n gpu_sparsekmeans python -m pip install -e .

conda install -n gpu_sparsekmeans -c conda-forge cupy  -y

conda run -n gpu_sparsekmeans python -m pip install matplotlib gdown

cd ../GPU_benchmark
gdown 1yLESCC_qqiI4EiM1Yjd7N_DBt6aQgWwh
unzip smalldataset.zip
rm smalldataset.zip

echo "To use this environment in your current shell:"
echo "  conda activate gpu_sparsekmeans"