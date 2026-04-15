# MANIFEST – Reproducibility Bundle for H-GEMO

## Source code (`hgemo/`)

| File               | Description                                          |
|--------------------|------------------------------------------------------|
| `__init__.py`      | Package docstring                                    |
| `cli.py`           | Command-line interface (`run` subcommand)            |
| `config.py`        | YAML config loading + dotted-key overrides           |
| `datasets.py`      | Dataset loading (pre-folded dense benchmark)         |
| `experiment.py`    | Main experiment runner (Algorithm 1 in the paper)    |
| `genotypes.py`     | Hypergraph & bitstring genotype operators            |
| `logging_utils.py` | Per-run file logger                                  |
| `metrics.py`       | Multi-label metrics, Pareto dominance, hypervolume   |
| `ml_eval.py`       | ML-kNN evaluator wrapper (tri-objective)             |
| `mlknn_impl.py`    | ML-kNN implementation (torch & sklearn backends)     |
| `npz_format.py`    | Sparse-matrix I/O helpers                            |
| `nsga2.py`         | NSGA-II engine (non-dominated sort, crowding, etc.)  |
| `utils.py`         | JSONL logger + seed utilities                        |

## Configuration (`configs/`)

| File                               | Description                                |
|------------------------------------|--------------------------------------------|
| `main_bench.yaml`                  | Main 20-dataset benchmark (hypergraph)     |
| `bitstring_ablation.yaml`          | Bitstring genotype ablation                |
| `smoke.yaml`                       | Quick smoke test (~200 evals, Emotions)    |
| `sensitivity/pop_size_20.yaml`     | HG population size = 20                    |
| `sensitivity/pop_size_30.yaml`     | HG population size = 30                    |
| `sensitivity/pop_size_80.yaml`     | HG population size = 80                    |
| `sensitivity/bs_pop20.yaml`        | Bitstring population size = 20             |
| `sensitivity/bs_pop30.yaml`        | Bitstring population size = 30             |
| `sensitivity/bs_pop50.yaml`        | Bitstring population size = 50             |
| `sensitivity/bs_pop80.yaml`        | Bitstring population size = 80             |
| `sensitivity/routing_*.yaml`       | Mutation routing variants (3 configs)      |
| `sensitivity/topm_ratio_*.yaml`    | Top-M ratio variants (4 configs)           |

## Scripts (`scripts/`)

| File                     | Description                                  |
|--------------------------|----------------------------------------------|
| `create_venv.sh`         | Create virtual environment + install deps    |
| `launch_bench.sh`        | Run main 20-dataset benchmark                |
| `launch_bitstring.sh`    | Run bitstring ablation                       |
| `launch_sensitivity.sh`  | Run all sensitivity analyses                 |
| `check_bundle.sh`        | Verify bundle integrity + optional smoke     |

## Metadata

| File               | Description                         |
|--------------------|-------------------------------------|
| `README.md`        | Full usage documentation            |
| `MANIFEST.md`      | This file                           |
| `pyproject.toml`   | Package metadata                    |
| `requirements.txt` | Python dependencies                 |
| `.gitignore`       | Git ignore rules                    |
