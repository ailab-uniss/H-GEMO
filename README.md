# H-GEMO

> **Hypergraph-Enhanced Multi-Objective Evolutionary Feature Selection
> for Multi-Label Classification**

H-GEMO is a novel **Multi-Objective Evolutionary Feature Selection (MLFS)** framework for **Multi-Label Classification** that models the feature search space using the expressive power of **hypergraphs**. 

In conventional feature selection, representing subsets of features via simple bitstrings fails to capture the complex, higher-order relationships between features and multiple labels. H-GEMO represents candidate feature subsets as hypergraphs, enabling the evolutionary algorithm to guide the search process more efficiently and robustly by understanding which groups of features jointly predict subsets of labels.

This repository serves as a highly optimized, fully functional, and extensible MLFS toolkit.

## ✨ Key Features
- **Hypergraph-based Evolutionary Search:** Uses hypergraph structures as genotypes to preserve, recombine, and mutate feature groups based on label correlations.
- **Multi-Objective Optimization:** Based on NSGA-II to simultaneously maximize predictive performance (e.g., Macro F1, Micro F1) while strongly minimizing the ratio of selected features.
- **GPU-Accelerated ML-kNN:** Features a highly optimized PyTorch implementation of the ML-kNN algorithm to drastically speed up fitness evaluations throughout the evolutionary process.
- **Configurable & Transparent:** Easily integrates with custom datasets, natively supports various similarity metrics, and allows deep configuration via YAML files.

---

## 🚀 Quick Start

```bash
# 1. Create a virtual environment and install dependencies
bash scripts/create_venv.sh
source .venv/bin/activate

# 2. Extract the 20 benchmark datasets (~22 MB zip → ~1 GB on disk)
bash scripts/prepare_datasets.sh

# 3. Verify the bundle is intact
bash scripts/check_bundle.sh

# 4. Run a quick smoke test (~2 min on GPU, uses Emotions)
bash scripts/check_bundle.sh --smoke

# 5. Run the full main benchmark (Table 1) on all 20 datasets
bash scripts/launch_bench.sh

# 6. Or run only on a subset of datasets:
bash scripts/launch_bench.sh Yeast Scene Emotions

# 7. Run the bitstring ablation (Table 2)
bash scripts/launch_bitstring.sh

# 8. Run all sensitivity analyses (Figs 4–6)
bash scripts/launch_sensitivity.sh
```

## Requirements

| Dependency        | Version  | Notes                            |
|-------------------|----------|----------------------------------|
| Python            | ≥ 3.10   |                                  |
| NumPy             | ≥ 1.24   |                                  |
| SciPy             | ≥ 1.10   |                                  |
| scikit-learn      | ≥ 1.2    |                                  |
| scikit-multilearn | ≥ 0.2    | For stratified label-set splits  |
| NetworkX          | ≥ 3.0    |                                  |
| PyYAML            | ≥ 6.0    |                                  |
| joblib            | ≥ 1.2    |                                  |
| PyTorch           | ≥ 2.0    | Optional: GPU-accelerated ML-kNN |

Install PyTorch separately from <https://pytorch.org/get-started/>.

## Dataset Preparation

All **20 benchmark datasets** used in the paper are shipped with this
repository as a single compressed archive:

```
data/dense_benchmark_v3/paper_datasets.zip      (~22 MB, 200 .npz files)
```

Extract them once with:

```bash
bash scripts/prepare_datasets.sh           # idempotent
bash scripts/prepare_datasets.sh --force   # re-extract / overwrite
```

This produces the layout the loader expects:

```
data/dense_benchmark_v3/<NAME>/
    fold0/  trainval.npz  test.npz
    fold1/  trainval.npz  test.npz
    ...
    fold4/  trainval.npz  test.npz
```

Each `.npz` file stores dense arrays `X` (float32) and `Y` (int8, binary
indicators).  The 5 folds are the exact stratified splits used to
produce all results in the paper (same random seed, same partitions).

The **Emotions** dataset is also kept extracted in-tree so the smoke
test can run before `prepare_datasets.sh` is invoked.

### The 20 datasets at a glance

| Dataset       | #Inst. | #Feat. | #Labels | Domain      |
|---------------|-------:|-------:|--------:|-------------|
| Arts          |  5000  |   462  |    26   | Text        |
| Business      |  5000  |   438  |    30   | Text        |
| Computers     |  5000  |   681  |    33   | Text        |
| Education     |  5000  |   550  |    33   | Text        |
| Emotions      |   593  |    72  |     6   | Music       |
| Enron         |  1702  |  1001  |    53   | Text        |
| Entertain     |  5000  |   640  |    21   | Text        |
| Foodtruck     |   407  |    32  |    12   | Survey      |
| Genbase       |   662  |  1185  |    27   | Biology     |
| Health        |  5000  |   612  |    32   | Text        |
| Medical       |   978  |  1449  |    45   | Medical     |
| Recreation    |  5000  |   606  |    22   | Text        |
| Reference     |  5000  |   793  |    33   | Text        |
| Scene         |  2407  |   294  |     6   | Vision      |
| Science       |  5000  |   743  |    40   | Text        |
| Slashdot      |  3782  |  1079  |    22   | Text        |
| Social        |  5000  |  1047  |    39   | Text        |
| Water-quality |  1060  |    16  |    14   | Environment |
| Yeast         |  2417  |   103  |    14   | Biology     |
| Yelp          | 10806  |   671  |     5   | Text        |

### Adding your own datasets

Place any extra dataset under `data/dense_benchmark_v3/<NAME>/` following
the same `foldK/{trainval,test}.npz` convention, then reference it by
name via `--override dataset.name=<NAME>` (see next section).

## Running Individual Experiments

Every run is launched through the same CLI; the dataset and the fold
index are selected from the command line, so **any of the 20 datasets**
(or any custom one you add) can be evaluated without editing a YAML
file:

```bash
# Generic template
python -m hgemo.cli run \
    --config configs/main_bench.yaml \
    --fold-idx <K>                       \  # K in {0,1,2,3,4}
    --override dataset.name=<NAME>       \
               logging.out_dir=runs/<my_run>
```

### Examples

```bash
# 1. Single fold of a single dataset
python -m hgemo.cli run --config configs/main_bench.yaml \
    --fold-idx 0 --override dataset.name=Yeast \
                            logging.out_dir=runs/manual/Yeast

# 2. Five folds of one dataset (full CV) – plain bash loop
for K in 0 1 2 3 4; do
    python -m hgemo.cli run --config configs/main_bench.yaml \
        --fold-idx $K --override dataset.name=Scene \
                                logging.out_dir=runs/manual/Scene
done

# 3. A subset of datasets via the helper script (5 folds each)
bash scripts/launch_bench.sh Yeast Scene Foodtruck

# 4. All 20 datasets × 5 folds (full Table 1 reproduction)
bash scripts/launch_bench.sh

# 5. Same dataset, custom hyper-parameters via further overrides
python -m hgemo.cli run --config configs/main_bench.yaml \
    --fold-idx 0 --override dataset.name=Medical \
                            evolution.pop_size=80 \
                            evolution.max_evals_per_feature=50 \
                            logging.out_dir=runs/exploration/Medical_pop80
```

The launch scripts (`launch_bench.sh`, `launch_bitstring.sh`,
`launch_sensitivity.sh`) are thin wrappers around the same CLI and accept
the dataset names as positional arguments — no source code edits required.

### CLI flags

| Flag         | Description                                     |
|--------------|-------------------------------------------------|
| `--config`   | Path to a YAML config file                      |
| `--fold-idx` | Cross-validation fold index (0–4)               |
| `--override`  | Dotted-key overrides (e.g. `evolution.pop_size=100`) |

### Outputs per run

Each `<out_dir>_fold<K>/` directory contains:

| File                      | Description                                   |
|---------------------------|-----------------------------------------------|
| `pareto_front.csv`        | Validation-set Pareto front (objectives)      |
| `pareto_front_real.csv`   | Real metric values (MacroF1, MicroF1, ratio)  |
| `pareto_front_test.csv`   | Test-set Pareto front                         |
| `pareto_val_solutions.json` | Per-solution details (val)                  |
| `test_selected_by_val.json` | Per-solution details (test)                 |
| `summary.json`            | Run summary (HV, generations, timing, etc.)   |
| `population_masks.npz`    | Binary feature masks for all Pareto solutions |
| `config_used.yaml`        | Exact config used for this run                |
| `run.log`                 | Detailed per-generation log                   |

## Configuration Reference

The YAML config mirrors the paper's Table of hyperparameters.  Key sections:

```yaml
evolution:
  pop_size: 50               # Population size (N)
  crossover_prob: 0.9        # Crossover probability (p_c)
  mutation_prob: 0.5         # Mutation probability (p_m)
  max_evals_per_feature: 100 # Budget = D × max_evals_per_feature
  genotype: hypergraph       # or "bitstring"
  early_stopping:
    enabled: true
    mode: window             # HV-window early stopping
    window: 10
    rel_tol: 0.002
    patience: 2

hypergraph:
  construction: multilayer_label
  fl_relevance: mi
  ll_similarity: jaccard
  fl_topm_ratio: 0.15
  fl_topm_min: 3
  init_edge_prob: 0.3
  edge_prune_prob: 0.1
  injection_prob: 0.3
  graph_contraction_threshold: 0.70
  target_feature_ratio: 0.25
  min_feature_ratio: 0.03
  mutation_routing_swap: 0.40
  mutation_routing_inject: 0.20
  feature_cluster_edges: auto
```

## Reproducing Paper Tables and Figures

| Paper element                          | Config                              | Script                        |
|----------------------------------------|-------------------------------------|-------------------------------|
| Multi-cap rank plot (fig:multicap_ranks) | `configs/main_bench.yaml`           | `scripts/launch_bench.sh`     |
| HV convergence (fig:hv_convergence)    | `configs/main_bench.yaml` + `configs/bitstring_ablation.yaml` | both launch scripts |
| Cap-sweep (fig:cap_sweep)              | `configs/main_bench.yaml` + `configs/bitstring_ablation.yaml` | both launch scripts |
| Sensitivity (fig:sensitivity)          | `configs/sensitivity/*.yaml`        | `scripts/launch_sensitivity.sh` |
| Bitstring ablation                     | `configs/bitstring_ablation.yaml`   | `scripts/launch_bitstring.sh` |

## Code Structure

```
repro_paper_runs/
├── hgemo/                  # Source code (13 files, ~2 800 lines)
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Config loading
│   ├── datasets.py         # Dataset I/O
│   ├── experiment.py       # Main experiment loop
│   ├── genotypes.py        # Hypergraph + bitstring genotypes
│   ├── metrics.py          # Multi-label metrics + HV
│   ├── ml_eval.py          # ML-kNN evaluator
│   ├── mlknn_impl.py       # ML-kNN implementation
│   ├── nsga2.py            # NSGA-II engine
│   └── …                   # Utilities
├── configs/                # YAML configs (17 files)
│   ├── main_bench.yaml
│   ├── bitstring_ablation.yaml
│   ├── smoke.yaml
│   └── sensitivity/        # 14 sensitivity variants
├── scripts/                # Shell launch scripts
│   ├── prepare_datasets.sh # Extract paper_datasets.zip
│   ├── create_venv.sh
│   ├── check_bundle.sh
│   ├── launch_bench.sh
│   ├── launch_bitstring.sh
│   └── launch_sensitivity.sh
├── data/                   # Datasets
│   └── dense_benchmark_v3/
│       ├── paper_datasets.zip   # 20 datasets, ~22 MB (extract with prepare_datasets.sh)
│       └── Emotions/            # pre-extracted for the smoke test
├── requirements.txt
├── pyproject.toml
├── MANIFEST.md
└── README.md               # This file
```

## License

MIT License.  See the main repository for full terms.

## Citation

If you use this code, please cite:

```bibtex
@article{hgemo2025,
  title   = {Hypergraph-Enhanced Multi-Objective Evolutionary Feature
             Selection for Multi-Label Classification},
  year    = {2025}
}
```
