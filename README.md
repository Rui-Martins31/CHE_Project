# CHE Project: HAR-kNN

Course project for **Efficient Heterogeneous Computing (M.EIC033)** at FEUP, 2026/27.
The goal is to accelerate the `knn_classifyinstance` function ([knn/src/knn.c](knn/src/knn.c)) on a multicore CPU (OpenMP) and on an ARM + FPGA SoC (Vitis HLS), while keeping the classification results exact.

## Repository layout

```
.
├── data/                 # datasets (NOT in git, see below)
│   ├── wisdm/
│   └── pamap2/
├── docs/                 # project statement and slides (PDF)
├── knn/                  # kNN C implementation
│   ├── Makefile
│   ├── src/
│   ├── bin/              # compiled binary (ignored)
│   ├── build/            # object files (ignored)
│   └── run/              # working dir for profiling runs (ignored)
├── scripts/              # helper scripts (benchmarking, profiling, plots, ...)
└── results/              # measurements and profiling reports
```

## Datasets

The dataset files (`*.dat`) are **not stored in this repository** because they are too large for GitHub.

### Where to get them

The `.dat` files ship with the **JACA Benchmark Suite (HAR-kNN)** that the course provides alongside the project statement (see Teams).
After downloading it, copy the files from each scenario folder into `data/`:

| In the JACA Benchmark Suite | In this repo |
|---|---|
| `sources/har-knn/scenario-wisdm/*.dat` | `data/wisdm/` |
| `sources/har-knn/scenario-pamap2/*.dat` | `data/pamap2/` |

For example:

```sh
JBS=/path/to/JACABenchSuite
cp "$JBS"/sources/har-knn/scenario-wisdm/*.dat  data/wisdm/
cp "$JBS"/sources/har-knn/scenario-pamap2/*.dat data/pamap2/
```

Each dataset folder should then contain:

| File | Purpose |
|---|---|
| `train1.dat`, `train2.dat`, `test1.dat`, `test2.dat` | Raw sensor data (HAR-kNN) |
| `train1_norm.dat`, `train2_norm.dat`, `train3.dat`, `train4.dat` | Training feature vectors (normalized) |
| `train1_notnorm.dat`, `train2_notnorm.dat` | Training feature vectors (not normalized) |
| `test1_notnorm.dat`, `test2_notnorm.dat` | Testing feature vectors |
| `min.dat`, `max.dat` | Normalization bounds |
| `key-READ1-k{3,5,7,20}.dat` | Expected classifications, used to verify results |

The `Readme.txt` in each dataset folder lists which files each `READ` mode uses.
The default build (`READ=4`) only needs `train2_norm.dat` and `test2_notnorm.dat`.

### Original data sources

The course staff derived the benchmark files from the public datasets below.
They are listed for reference only: the code needs the course's preprocessed `.dat` files, not the raw downloads.

- **WISDM** v1.1: Wireless Sensor Data Mining Lab, Fordham University, <https://www.cis.fordham.edu/wisdm/>.
  J. R. Kwapisz, G. M. Weiss, S. A. Moore, "Activity recognition using cell phone accelerometers", *SIGKDD Explor. Newsl.* 12(2), 2010. <https://doi.org/10.1145/1964897.1964918>
- **PAMAP2** Physical Activity Monitoring: <https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring>.
  A. Reiss, D. Stricker, "Introducing a New Benchmarked Dataset for Activity Monitoring", ISWC 2012. <https://doi.org/10.1109/ISWC.2012.13>.
  Feature extraction follows P. J. S. Ferreira, J. Mendes Moreira, J. M. P. Cardoso, iWOAR '23. <https://doi.org/10.1145/3615834.3615851>

## Building and running

Build from the `knn/` folder.
The Makefile finds the datasets in `../data` on its own, so `JBSPATH` from the course slides is **not** needed.

```sh
cd knn
make run SCENARIO=wisdm            # default scenario
make run SCENARIO=pamap2
make run SCENARIO=wisdm K=5 DT=2   # K = 3/5/7/20, DT: 1 = double, 2 = float
```

Other variables you can override: `READ`, `SPECIALIZED`, `USE_SQRT`, `OPT` (e.g. `OPT=-O3`), and `DATA_DIR` (to use datasets stored elsewhere).
Makefile targets: `clean`, `cleanall`, `comp`, `run`, `comprofiling`, `runprofiling`.
