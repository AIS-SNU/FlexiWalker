# FlexiWalker

**FlexiWalker** (EuroSys '26) is a GPU framework that delivers efficient, workload-generic support for *dynamic* random walks — walks whose transition probabilities depend on runtime state. It combines:

1. **High-performance rejection-sampling and reservoir-sampling kernels** that eliminate global reductions, redundant memory accesses, and unnecessary RNG calls — chosen via a design-space study that identified these two techniques as the most amenable to massive GPU parallelism.
2. **A lightweight first-order cost model** that picks the faster of the two kernels per node at runtime, since neither dominates across all workloads.
3. **A compile-time component** that takes a user-supplied walker class (a C++ description of the sampling distribution), analyzes it with Clang+LLVM, and automatically specializes it into the optimized sampling building blocks above.


## Requirements

- NVIDIA GPU with a CUDA 12.1-compatible driver
- CUDA ≥ 11.6 (tested on 12.1)
- CMake ≥ 3.15
- g++ supporting C++17 with OpenMP (tested with g++ 11 on Ubuntu 22.04)
- gflags ≥ 2.2 (`apt-get install libgflags-dev`)
- LLVM/Clang 17 (for the compilation pipeline)
- nlohmann_json
- Python 3 with `jinja2`, `numpy` (and `pytest`, `pytest-xdist` for the test suite)
- Docker (recommended — provides a reproducible toolchain)

## Toolchain container (recommended)

The reproducible toolchain (CUDA 12.1, Clang/LLVM 17, gflags, nlohmann_json) lives in [`docker/Dockerfile`](docker/Dockerfile). Build the image once, then run the pipeline and build inside it:

```bash
docker build -t flexiwalker -f docker/Dockerfile .
export FLEXIWALKER_DATA="$HOME/flexiwalker_data"   # host path mounted as /data inside the container

docker run --rm -it --gpus all \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v "$PWD":/flexiwalker -v "$FLEXIWALKER_DATA":/data \
  -w /flexiwalker flexiwalker bash
```

All commands below are run **inside that container shell** (or directly on the host if you have an equivalent toolchain).

## Run the pipeline

The pipeline analyzes the walker classes in `include/app.cuh` and emits the per-walker GPU kernels in `include/generated/*.cuh` that the build then compiles. **Run it before building**, and again every time you modify `include/app.cuh`:

```bash
python3 run_pipeline.py
```

The pipeline runs seven stages (dummy generation → metadata extraction → template generation → LLVM analysis → type analysis → graph-field detection → code generation).

## Build

```bash
mkdir -p build && cd build && cmake .. && make -j
```

This produces `./build/bin/flowwalker`. Each of the five shipped walkers (Node2vec, Node2vec_weighted, Metapath, Metapath_weighted, PPR_second) is compiled into the binary with its pipeline-generated kernel.

## Prepare datasets

`data/get_dataset.sh <download_dir>` downloads all datasets used in the paper. The SNAP datasets ship as `.txt.gz` (gunzip → ASCII edgelist). The LAW datasets ship as WebGraph `.graph` (convert to ASCII via WebGraph's `ArcListASCIIGraph` — see [`data/README.md`](data/README.md)). Then compile and run `data/EdgeListToCSR.cpp` to produce `<name>_xadj.bin`, `<name>_edge.bin`, `<name>_label.bin`, `<name>_weight.bin`. See [`data/README.md`](data/README.md) for full details and the binary file format.

Per-dataset paths live in `config/graphs/<name>.config` and are read at runtime — edit them to point at your own files without recompiling. Adding new graph fields in `config/graph_fields.config`, however, requires re-running the pipeline.

## Quick example

Run the five pipeline-generated walkers (Node2vec, Node2vec_weighted, Metapath, Metapath_weighted, PPR_second) against one dataset:

```bash
scripts/run_templ_one.sh 0 com-youtube   # <gpu_id> <dataset_name>
```

To sweep all datasets from the paper, use `scripts/run_templ.sh <gpu_id>`.

Dataset binaries must be prepared separately and laid out as `data/<name>/<name>_*.bin`. When running in Docker, set `FLEXIWALKER_DATA` to a host directory with that layout — the `-v "$FLEXIWALKER_DATA":/data` mount in the toolchain command above makes it available at `/data` inside the container, and the shipped configs in `config/graphs/` resolve to it. See [`data/README.md`](data/README.md) for file format. We do not ship the preprocessed graph binaries.

## Add your own walker

See [`docs/WALKER_API.md`](docs/WALKER_API.md) for the walker-class API, and [`docs/WALKER_TEMPLATE.cuh`](docs/WALKER_TEMPLATE.cuh) for a copy-paste starter. The compiler contract (how `graph_fields.config` drives `_MAX`/`_MIN`/`_SUM` aggregates, and when the analyzer falls back to eRVS-only) is documented in `docs/WALKER_API.md`.

## Repository layout

```
├── README.md                   # this file
├── LICENSE
├── CMakeLists.txt
├── run_pipeline.py             # pipeline entrypoint
├── docs/                       # API reference + developer guide
├── scripts/                    # reproduction helpers (dataset sweep)
├── config/                     # graph_fields.config + per-dataset configs
├── data/                       # tiny example + CSR conversion tools
├── include/                    # walker runtime + app.cuh
│   └── generated/              # pipeline output (regenerated)
├── src/                        # main.cu, walk.cu
├── pipeline/                   # Python pipeline framework
├── tools/                      # Clang + LLVM analysis tools, codegen templates
├── tests/                      # compiler + e2e test harness
├── docker/                     # reproducible toolchain image
└── lint/
```

## Roadmap

Planned improvements:

- **Auto-generated walker dispatch.** Eliminate the manual flag/parser/dispatch-chain edits in `src/main.cu` and `src/walk.cu` for new walkers — the pipeline will generate them from `LLVM_CTOR` parameter annotations.

## Acknowledgment

FlexiWalker builds on **FlowWalker** (Mei et al., VLDB 2024) — the GPU random-walk runtime. The FlowWalker authors open-sourced their work at <https://github.com/junyimei/flowwalker-artifact>, and we thank them for enabling this research.

## Citation

(EuroSys '26 citation TBA.)

```bibtex
@article{park2025flexiwalker,
  title={FlexiWalker: Extensible GPU Framework for Efficient Dynamic Random Walks with Runtime Adaptation},
  author={Park, Seongyeon and Song, Jaeyong and Shin, Changmin and Kim, Sukjin and Hong, Junguk and Lee, Jinho},
  journal={arXiv preprint arXiv:2512.00705},
  year={2025}
}

```

## License

See [`LICENSE`](LICENSE).
