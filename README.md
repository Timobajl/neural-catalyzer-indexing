# Neural Catalyzer for High-Dimensional Similarity Search Indexing

Code and results of the bachelor's thesis *Neural Catalyzer for High-Dimensional Similarity
Search Indexing* (Timotej Martykán, Faculty of Informatics, Masaryk University, 2026).

The thesis tests whether the Neural Catalyzer of Sablayrolles et al. (ICLR 2019), used as a
continuous preprocessor without its discretization step, improves five similarity-search
indexes: IVFFlat, HNSW, the Learned Metric Index (LMI), NeuralLSH and the Neural Partitioner.
The datasets are BigANN-1M, Deep1B-1M and the 100M subset of LAION-5B from the SISAP 2024
Indexing Challenge.

## Repository layout

| Folder | Content |
|---|---|
| `catalyzer/` | Neural Catalyzer (copy of `spreadingvectors`) with the changes of this thesis, training and export scripts |
| `indexes/ivfflat/` | IVFFlat benchmark (FAISS) and its job scripts |
| `indexes/hnsw/` | HNSW benchmark (FAISS) and its job scripts |
| `indexes/lmi/` | Learned Metric Index (copy) with `lmi_index.py` and job scripts |
| `indexes/neural-lsh/` | NeuralLSH (copy of `learn-to-hash`) with `nlsh_index.py`, job scripts and `KaHIP/` |
| `indexes/neural-partitioner/` | Neural Partitioner (copy) with `np_index.py` and job scripts |
| `other-experiments/` | Ceilings (exhaustive search), control run, LAION duplicates, catalyzer inference time, LAION consistency cache, FAISS vs. `dist_rank` check |
| `plots/` | Scripts for the figures and the build-time table of the thesis |
| `results/` | Result CSV files used in the thesis |

Each copied repository keeps its original files. Its folder holds a patch file with every
change made in this thesis against the original commit (see [Copied code](#copied-code)).

## Requirements

All results were produced in one conda environment. `environment-explicit.txt` lists all of its
209 packages with their exact builds, plus seaborn for the figure scripts (linux-64, about 3.6 GB
to download), and recreates it:

```bash
conda create --name neural-catalyzer-indexing --file environment-explicit.txt
```

`environment.yml` names only the main packages, in the same versions, for systems where the
exact list cannot be used:

```bash
conda env create -f environment.yml
```

Main packages:

| Package | Version |
|---|---|
| Python | 3.10.20 |
| NumPy | 1.26.4 |
| FAISS | 1.11.0 (GPU build) |
| PyTorch | 2.5.1 with CUDA 11.8 |
| h5py | 3.16.0 |
| pandas | 2.3.3 |
| SciPy | 1.15.2 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 (figures only) |

NeuralLSH needs the KaHIP graph partitioner. Build it once (CMake and a C++ compiler):

```bash
cd indexes/neural-lsh/KaHIP && ./compile_withcmake.sh
```

The binary `indexes/neural-lsh/KaHIP/deploy/kaffpa` is found through `kahip_dir=KaHIP` in
`indexes/neural-lsh/config`, so NeuralLSH runs from `indexes/neural-lsh/`.

## Data

| Dataset | Files | Source |
|---|---|---|
| BigANN | `bigann_base.bvecs`, `bigann_learn.bvecs`, `bigann_query.bvecs`, `gnd/idx_1M.ivecs` | `ftp://ftp.irisa.fr/local/texmex/corpus` |
| Deep1B | `base.1B.fbin`, `learn.350M.fbin`, `query.public.10K.fbin`, `deep_new_groundtruth.public.10K.bin` | Yandex, links below |
| LAION | `laion2B-en-clip768v2-n=100M.h5`, `laion2B-en-clip768v2-n=10M.h5`, `public-queries-2024-laion2B-en-clip768v2-n=10k.h5`, `gold-standard-dbsize=100M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5` | SISAP 2024 Indexing Challenge, https://huggingface.co/datasets/Coda-Research-Group/SISAP_2024_Indexing_Challenge |

Deep1B files:

- https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
- https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/learn.350M.fbin
- https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin
- https://storage.yandexcloud.net/yandex-research/ann-datasets/deep_new_groundtruth.public.10K.bin

The export scripts write the vectors used by every index into one data folder:

```
data/<dataset>/original/                       xb.npy, xq.npy, gt.npy, xt.npy
data/<dataset>/catalyzer_d<dout>_l<lambda>/    xbt.npy, xqt.npy
```

The export computes the Deep1B ground truth for the first million vectors by exact search
for the 10 000 public queries.

## Running the experiments

The job scripts are PBS scripts for the MetaCentrum grid (`qsub <script>`). Each script sets
`REPO_DIR` and the paths to the data, the raw datasets and the conda environment at its top.
Change them there, or pass the repository path with `qsub -v REPO_DIR=<path> <script>`.

1. **Train the catalyzers:** `catalyzer/train_scripts/model_train*.sh`. Checkpoints go to
   `catalyzer/checkpoints/<dataset><dout>/<lambda>/checkpoint.pth.best`.
2. **Export the vectors:** `catalyzer/export_scripts/export_run*.sh`, layout above.
3. **Ceilings and the choice of lambda:** `other-experiments/knn_scripts/knn_<dataset>.sh`.
4. **Indexes:** `indexes/<index>/<index>_scripts/*.sh`. Results go to `results/<index>/<dataset>/`.
5. **Other experiments:** the scripts in `other-experiments/*_scripts/`, plus
   `compute_true_neighbors.sh` (needed before the LAION index runs).
6. **Figures:** `python plots/generate_<name>.py` writes to `plots/output/`.
   `python plots/extract_build_time_table.py` prints the build-time table.

The reproduction of the original paper with the authors' checkpoints is
`catalyzer/reproduce.sh` (upstream) and `catalyzer/reproduce_metacentrum.sh`.

## Quick example

`example/quick_example.sh` runs the whole pipeline on a small part of Deep1B. It downloads the
first 100 000 base vectors, the first 200 000 learn vectors and the 10 000 queries (about 120 MB),
trains one catalyzer (64D, lambda 0.0002) for 10 epochs on 50 000 vectors, exports the vectors
and runs IVFFlat in the original and in the transformed space. It takes under 10 minutes
on a laptop CPU with 8 threads.

```bash
conda activate neural-catalyzer-indexing
bash example/quick_example.sh
```

IVFFlat prints the recall and the number of distance computations for every nprobe, and the
results are saved in `example/output/results/`. The numbers differ from the thesis, which uses
1M base vectors, 500 000 training vectors and 160 epochs.

## Results and the thesis

| Thesis | Script | Results |
|---|---|---|
| Fig. 5.1, Table 5.4 | `plots/generate_lambda_ceiling_figure.py` | `results/knn/*/knn_ceiling_results.csv`, `results/knn/laion/knn_ceiling_full.csv` |
| Fig. 5.2 | `plots/generate_pca_baseline_figure.py` | `results/knn/bigann/knn_ceiling_results_pca.csv` |
| Table 5.3 | `other-experiments/knn_authors.py` | `results/knn/*/knn_ceiling_authors.csv` |
| Figs. 5.3 to 5.11, C.1 to C.10 | `plots/generate_<index>_figures.py`, `plots/generate_knn_consistency_bigann.py` | `results/<index>/<dataset>/` |
| Table 5.5 | (bucket imbalance column) | `results/<index>/<dataset>/knn_consistency_*.csv` |
| Figs. 5.12, D.1, D.2, Table 5.6 | `plots/generate_spearman_figure.py` | `results/<index>/<dataset>/` |
| Table 5.7 | `other-experiments/rerank_original.py` | `results/rerank/bigann/` |
| Table 5.8 | `plots/extract_build_time_table.py` | `results/<index>/<dataset>/` |
| Table 5.9 | `other-experiments/measure_inference.py` | `results/inference/inference_times.csv` |
| Figs. 5.13, C.11, C.12 | `plots/generate_e2e_figure.py` | `results/<index>/<dataset>/`, `results/inference/` |
| Section 5.8 (LAION duplicates), Section 4.3 (training slice) | `other-experiments/count_duplicates.py`, `check_train_overlap.py` | `results/duplicates/laion/` |
| Section 5.10 (300 epochs) | `catalyzer/train_scripts/model_train_300.sh` | `results/knn/bigann/knn_ceiling_300.csv` |
| Table A.1 | `other-experiments/verify_knn_equivalence.py` | `results/knn_equivalence/verify_knn_results.md` |
| Appendix B | `indexes/neural-partitioner/np_scripts/eta/` | `results/np/eta/` |

Notes:

- **[to update after the repeated 1M runs]** Every build and search time is a single
  measurement. The LAION jobs ran on different nodes.

## Copied code

Each section names the source, the original commit and the patch file with all changes.
A patch applies with `git apply <patch>` at the root of a clone of the original repository at
the named commit. In this repository, `git apply -R --directory=<folder> <folder>/<patch>` from
the repository root restores the original files, for example
`git apply -R --directory=catalyzer catalyzer/catalyzer.patch`.

### Neural Catalyzer (`catalyzer/`)

Source: https://github.com/facebookresearch/spreadingvectors, commit
`7aae7398e9683fc26d7fb2bbeb15eb44bfee8021` (2021-06-11). Changes: `catalyzer/catalyzer.patch`.

- Data loading (`lib/data.py`): Deep1B from the `.fbin` files, LAION from HDF5 in chunks,
  dataset folders from `BIGANN_PATH`, `DEEP1B_PATH` and `LAION_PATH`.
- `train.py` accepts `--database laion` and prints the device. `eval.py` loads checkpoints on
  the CPU and times only encoding and evaluation.
- The discretization step is not used: training runs with the existing options
  `--validation_quantizers none --save_best_criterion none,rank=10`, and
  `export_vectors.py` writes the continuous output of the network.
- New files: `export_vectors.py`, `export_scripts/`, `train_scripts/`, `reproduce_metacentrum.sh`.

### Learned Metric Index (`indexes/lmi/`)

Source: https://github.com/Coda-Research-Group/LearnedMetricIndex, branch
`paper-sisap24-indexing-challenge`, commit `1163d64f9c0fd3e2cfb333267175ee4b55aa2ff1`
(2025-03-23). Changes: `indexes/lmi/lmi.patch`. `indexes/lmi/lmi_index_vs_task1.diff` shows
how `lmi_index.py` differs from `task1.py`.

- `lmi_index.py` is `task1.py` of the reference with the same network, training and search.
  Changes: `.npy` input, Euclidean or inner product, training sample without replacement,
  buckets in the input dtype (float16 only for LAION), small buckets return only their own
  vectors, a grid over the number of buckets and nprobe, k-NN consistency, distance-computation
  count, configurable threads, optional `--model-dir` (not used by the jobs).
- New files: `lmi_index.py`, `lmi_scripts/`, `LICENSE` from the main branch.

### NeuralLSH (`indexes/neural-lsh/`)

Source: https://github.com/twistedcubic/learn-to-hash, commit
`daaa2adfdc2fdcc981043c4f4511ce83cb8a4746` (2021-01-20). Changes:
`indexes/neural-lsh/neural-lsh.patch`. KaHIP: https://github.com/KaHIP/KaHIP, commit
`5935f349f65f1788a9b68fcf6d853e698d86956d` (2026-04-15), unchanged.

- `nlsh_index.py` runs the reference training and adds the search, the distance-computation
  count and the k-NN consistency. The k-NN graph comes from FAISS exact search instead of
  `dist_rank`, which works only with Euclidean distance. On LAION 100M it trains on 1M sampled
  vectors with k-means at the bottom level.
- Settings from the paper: its network for all datasets, 18 epochs on both levels, soft labels
  from 16 neighbors.
- Inner product for LAION: spherical k-means and ranking by inner product.
- Fixes needed to run it: debug stops, an undefined variable, bucket IDs reused after an empty
  class, a CPU and CUDA mismatch.
- `k_graph` for the graph, skips the reference's evaluation, file names by dataset and label,
  `config`: `kahip_dir=KaHIP`.
- New files: `nlsh_index.py`, `nlsh_scripts/`.

### Neural Partitioner (`indexes/neural-partitioner/`)

Source: https://github.com/abrar-fahim/Neural-Partitioner, commit
`3d8d817c0dab06e34a991923a27d5afd7854d2af` (2022-06-17). Changes:
`indexes/neural-partitioner/neural-partitioner.patch`.

- `np_index.py` replaces `main.py`: one model instead of an ensemble, the search, the
  distance-computation count and the k-NN consistency. On LAION 100M it trains on 1M sampled
  vectors and assigns every vector with the trained model. Models, the k'-NN matrix and the
  bin assignments are reused when their files exist.
- Settings from the paper: eta is a parameter, 100 epochs at the root (200 below, as in the
  reference), each node trains on batches of 4 % of its vectors, at least 2 048.
- Neighbors come from FAISS exact search on the raw vectors, in Euclidean distance or
  inner product.
- Fixes needed to run it: missing `org_n` argument, scaler saved with the model, CPU support,
  no `plt.show()`.
- The scaler runs on the GPU, fewer progress prints.
- New files: `np_index.py`, `np_dataset.py`, `np_scripts/` (`eta/` holds the eta sweep).

## License

The code written for this thesis and the changes in the patch files are released under the
GNU Lesser General Public License, version 3 (`COPYING.LESSER`, which builds on the GNU GPL
version 3 in `COPYING`). Everything else keeps the license of its original repository.
