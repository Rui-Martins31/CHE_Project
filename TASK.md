# Tasks

Checklist of the project steps from the statement ([docs/Project-kNN-M.EIC-CHE-2026-27.pdf](docs/Project-kNN-M.EIC-CHE-2026-27.pdf)).
Mark a task as done by changing `- [ ]` to `- [x]`.

Rules that apply to every step:

- The kNN must stay **exact**: classifications must match the original code (check against `key-READ1-k*.dat`).
- Streaming scenario: instances to classify arrive one at a time.
- Report the important experiments even when they do not improve time, power or energy.

## Setup

- [x] Reorganize the repository layout (`knn/`, `data/`, `docs/`, `scripts/`, `results/`)
- [x] Makefile finds datasets in `../data` (no `JBSPATH` needed)
- [x] Document where to get the datasets ([README.md](README.md))
- [x] Script to run gprof and gcov ([scripts/profile_gnu.py](scripts/profile_gnu.py))
- [x] Script to run VTune and Advisor ([scripts/profile_intel.py](scripts/profile_intel.py))

## Checkpoint 1: multicore CPU (PC/desktop/laptop)

Use both scenarios: WISDM and PAMAP2 (K = 3, 5, 20; float and double).

### 1.1 Profile the original program

- [x] gprof: flat profile and call counts, 12 configurations ([results/README.md](results/README.md))
- [x] gcov: line execution counts, 12 configurations ([results/README.md](results/README.md))
- [x] Identify the hotspots (distance loop in `get_*_NN`; `select_k_nearest` for large k)
- [x] Call graph (text version in [results/README.md](results/README.md))
- [x] Intel VTune: hotspots analysis (compare with gprof) ([results/README.md](results/README.md))
- [ ] Intel Advisor: roofline of the original code (empty on Windows/MinGW, rerun under WSL; see [results/README.md](results/README.md))
- [ ] Call graph as a figure for the report, decorated with calls and % time
- [ ] Draw the task graph, decorated with data sizes and dependencies
- [ ] Select and justify the code regions to migrate to a hardware accelerator

### 1.2 Characterize code versions with gcc flags

- [ ] Describe the experimental setup (CPU, RAM, OS, gcc version, power measurement tool)
- [ ] Choose the gcc flag sets to compare (e.g. `-O0`, `-O2`, `-O3`, `-Ofast`, `-march=native`, ...)
- [ ] Choose the code variants to compare (`DT`, `SPECIALIZED`, `USE_SQRT`, `K`, ...)
- [ ] Measure execution time for each version
- [ ] Measure power dissipation for each version
- [ ] Measure energy consumption for each version
- [ ] Compute the gains over the original code and plot them

### 1.3 Improve CPU performance

- [ ] Code transformations in `knn_classifyinstance` (and helpers, if useful)
- [ ] Parallelization with OpenMP
- [ ] Verify results stay exact after each change
- [ ] Use VTune and Advisor on each new version
- [ ] Show the position of each version in the Roofline model
- [ ] Fill the final table: throughput (inferences/s) and energy efficiency (inferences/mJ) for each scenario and K

## Checkpoint 2: ARM + FPGA SoC (baseline)

Scenario: WISDM, K = 3, float.

### 2.1 Offload to the FPGA

- [ ] Analyze alternatives to offload computations to the FPGA
- [ ] Characterize and optimize the selected code with Vitis HLS
- [ ] 2.1.1 Report FPGA resources used, maximum clock frequency, and latency (clock cycles)
- [ ] 2.1.2 Report execution times: SW only (ARM CPU) and HW/SW (ARM CPU + FPGA)

### 2.2 Evaluate the system

- [ ] Overall speedup: CPU + accelerator vs. CPU only

## Checkpoint 3: FPGA optimizations

Scenario: WISDM, K = 3, float.

- [ ] 3.1 Report the selected versions to evaluate
- [ ] 3.2 Analyze the optimizations and discuss possible extensions
- [ ] 3.3 Compare the SW version on the CPU with the HW/SW version (CPU + FPGA)

## Final deliverables

- [ ] Technical report
  - [ ] Experimental setup described in detail
  - [ ] Results and their analysis
  - [ ] Optimizations not evaluated but expected to have significant impact
- [ ] Presentation and discussion
