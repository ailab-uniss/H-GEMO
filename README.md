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

# 2. Verify the bundle is intact
bash scripts/check_bundle.sh

# 3. Run a quick smoke test (~2 min on GPU)
bash scripts/check_bundle.sh --smoke

# 4. Run the full main benchmark (Table 1)
bash scripts/launch_bench.sh

# 5. Run the bitstring ablation (Table 2)
bash scripts/launch_bitstring.sh

# 6. Run all sensitivity analyses (Figs 4–6)
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

The experiments expect pre-folded datasets in `data/dense_benchmark_v3/`.
Each dataset folder contains:

```
data/dense_benchmark_v3/<NAME>/
    fold0/  trainval.npz  test.npz
    fold1/  ...
    ...
    fold4/  ...
```

Each `.npz` file stores dense arrays `X` (float32) and `Y` (int8, binary
indicators).  The **Emotions** dataset (72 features, 6 labels, 593 samples)
is included in this bundle for immediate smoke-testing; see the paper's
Section 4.1 for the full list of 20 benchmark datasets and their sources.

To add additional datasets, place them in `data/dense_benchmark_v3/<NAME>/`
following the same fold structure.  If the bundle lives inside the main
project tree, a symlink also works:
```bash
cd repro_paper_runs && ln -sfn ../data data
```

## Running Individual Experiments

Every run is launched through the CLI:

```bash
python -m hgemo.cli run \
    --config configs/main_bench.yaml \
    --fold-idx 0 \
    --override dataset.name=Emotions logging.out_dir=runs/my_test
```

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
├── data/                   # Datasets (not included, see above)
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
