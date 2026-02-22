# XCT
python3 -u memory_usage.py --dataset eurlex \
                            --n_clusters 100

python3 -u memory_usage.py --dataset eurlex \
                            --n_clusters 500

python3 -u memory_usage.py --dataset wiki10_31k \
                            --n_clusters 100

python3 -u memory_usage.py --dataset wiki10_31k \
                            --n_clusters 500

# BX (SpMM)
python3 -u memory_usageBX.py --dataset eurlex \
                            --n_clusters 100

python3 -u memory_usageBX.py --dataset eurlex \
                            --n_clusters 500

python3 -u memory_usageBX.py --dataset wiki10_31k \
                            --n_clusters 100

python3 -u memory_usageBX.py --dataset wiki10_31k \
                            --n_clusters 500

# BX (SpGEMM)
python3 -u memory_usageBX_SpGEMM.py --dataset eurlex \
                            --n_clusters 100

python3 -u memory_usageBX_SpGEMM.py --dataset eurlex \
                            --n_clusters 500

python3 -u memory_usageBX_SpGEMM.py --dataset wiki10_31k \
                            --n_clusters 100

python3 -u memory_usageBX_SpGEMM.py --dataset wiki10_31k \
                            --n_clusters 500