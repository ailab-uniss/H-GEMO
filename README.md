# H-GEMO

> **H-GEMO: Template-Based Hypergraph Genotypes for Multi-Objective
> Multi-Label Feature Selection**

H-GEMO is a multi-objective framework for **multi-label feature selection
(MLFS)**. Candidate feature subsets are encoded as **template-based
hypergraph genotypes**: hyperedges group features that are jointly
relevant to subsets of correlated labels, and the evolutionary operators
recombine, mutate and repair these groups instead of individual bits.

- **Hypergraph genotype:** a fixed template containing label-centred
  hyperedges derived from mutual information and feature-cluster hyperedges;
  each individual activates and refines a subset of these groups.
- **Multi-objective search:** NSGA-II minimises (1 − Macro-F1),
  (1 − Micro-F1) and the fraction of selected features.
- **ML-kNN wrapper:** fitness is computed with ML-kNN (k = 5, s = 1),
  with an optional PyTorch/GPU implementation for fast evaluation.

This repository contains the H-GEMO implementation and one example
dataset (**Emotions**, 5 pre-computed folds).

---

## Quick start

```bash
# 1. Create a virtual environment and install dependencies
bash scripts/create_venv.sh
source .venv/bin/activate

# 2. Check the installation (add --demo for a ~10 s run on Emotions)
bash scripts/check_bundle.sh --demo

# 3. Run H-GEMO with the default settings on one fold of Emotions
python -m hgemo.cli run --config configs/hgemo.yaml --fold-idx 0 \
    --override dataset.name=Emotions logging.out_dir=runs/Emotions
```

## Requirements

| Dependency        | Version  | Notes                            |
|-------------------|----------|----------------------------------|
| Python            | ≥ 3.10   |                                  |
| NumPy             | ≥ 1.24   |                                  |
| SciPy             | ≥ 1.10   |                                  |
| scikit-learn      | ≥ 1.2    |                                  |
| scikit-multilearn | ≥ 0.2    | Stratified train/validation split |
| NetworkX          | ≥ 3.0    |                                  |
| PyYAML            | ≥ 6.0    |                                  |
| joblib            | ≥ 1.2    |                                  |
| PyTorch           | ≥ 2.0    | Optional: GPU-accelerated ML-kNN |

Install PyTorch separately from <https://pytorch.org/get-started/>.
Without PyTorch, ML-kNN falls back to scikit-learn on the CPU.

## Data format

Datasets live under `data/<NAME>/`, one folder per cross-validation fold:

```
data/<NAME>/
    fold0/  trainval.npz  test.npz
    ...
    fold4/  trainval.npz  test.npz
```

Each `.npz` file stores dense arrays `X` (float32, instances × features)
and `Y` (integer, instances × labels, binary indicators). The `trainval`
part is split internally into training and validation sets with
iterative stratification; `test` is used only to evaluate the final
solutions.

The repository ships `data/Emotions/` (593 instances, 72 features,
6 labels). To use your own data, create `data/<NAME>/` with the same
layout and pass `--override dataset.name=<NAME>`.

## Running H-GEMO

```bash
python -m hgemo.cli run \
    --config configs/hgemo.yaml \
    --fold-idx <K> \
    --override dataset.name=<NAME> logging.out_dir=runs/<NAME>
```

Full 5-fold cross-validation:

```bash
for K in 0 1 2 3 4; do
    python -m hgemo.cli run --config configs/hgemo.yaml --fold-idx $K \
        --override dataset.name=Emotions logging.out_dir=runs/Emotions
done
```

Any configuration value can be changed from the command line with dotted
keys, e.g. `--override evolution.pop_size=80 model.mlknn_device=cpu`.

| Flag         | Description                                          |
|--------------|------------------------------------------------------|
| `--config`   | YAML configuration file                              |
| `--fold-idx` | Cross-validation fold index (0–4)                    |
| `--override` | Dotted-key overrides (e.g. `evolution.pop_size=100`) |

### Outputs

Each run writes `<out_dir>_fold<K>/` with:

| File                        | Description                                   |
|-----------------------------|-----------------------------------------------|
| `pareto_front.csv`          | Validation-set Pareto front (objectives)      |
| `pareto_front_real.csv`     | Metric values (Macro-F1, Micro-F1, ratio)     |
| `pareto_front_test.csv`     | Test-set values of the same solutions         |
| `pareto_val_solutions.json` | Per-solution details (validation)             |
| `test_selected_by_val.json` | Per-solution details (test)                   |
| `summary.json`              | Run summary (exact HV, generations, timing, …) |
| `population_masks.npz`      | Binary feature masks of the Pareto solutions  |
| `config_used.yaml`          | Exact configuration used                      |
| `run.log`                   | Per-generation log                            |

## Configuration

`configs/hgemo.yaml` holds the default settings; `configs/demo.yaml` is
a tiny-budget version for a quick check. Main parameters:

```yaml
evolution:
  pop_size: 50               # population size
  crossover_prob: 0.9
  mutation_prob: 0.5
  max_evals_per_feature: 100 # evaluation budget = D × 100
  early_stopping:
    enabled: true
    mode: paper              # reproduce the dataset-specific paper protocol
    window: 10
    rel_tol: 0.002
    patience: 2

hypergraph:
  construction: multilayer_label
  fl_relevance: mi           # feature–label relevance
  fl_mi_estimator: auto      # deterministic policy described in the paper
  ll_similarity: jaccard     # label–label similarity
  fl_topk_labels_per_feature: 8 # mutual-filter rank per feature
  fl_topm_ratio: 0.15        # fraction of features per label hyperedge
  fl_topm_min: 3             # minimum candidate-set size
  fl_mutual_floor: 3         # candidates always retained per label
  init_edge_prob: auto       # 0.3 + 0.3 exp(-D/50)
  edge_prune_prob: 0.1
  edge_toggle_prob: 0.5
  injection_k: 1
  graph_contraction_threshold: 0.70
  target_feature_ratio: 0.25
  min_feature_ratio: 0.03
  mutation_routing_swap: 0.40
  mutation_routing_inject: 0.20
  feature_cluster_edges: auto
```

The automatic MI policy binarises features into presence/absence when at
least 25% of them have occupancy below 0.5; otherwise it treats the observed
values as discrete categories. The optional continuous kNN estimator can be
selected explicitly, but it is not the policy used for the reported benchmark.
The search seed is independent of `dataset.split_seed`, so repeated optimizer
runs use the same inner train/validation partition.

Early stopping monitors the additive proxy defined in the paper, not exact
hypervolume. With `mode: paper`, the proxy and the quality index `q` must both
stagnate for Arts, Education, Foodtruck, Health, Recreation, Reference,
Science, Social and Yelp; the other datasets use the proxy-only window rule.
Exact hypervolume with reference point `(1, 1, 1)` is computed only for the
final summary. Set `early_stopping.enabled: false` for the fixed-budget
protocol used in the controlled evolutionary comparisons.

## Code structure

```
hgemo/
├── cli.py            # command-line entry point
├── config.py         # YAML loading and dotted-key overrides
├── datasets.py       # dataset loading and train/validation split
├── experiment.py     # main loop for one fold
├── genotypes.py      # hypergraph template, genome, crossover, mutation, repair
├── metrics.py        # metrics, Pareto dominance, stopping proxy, exact HV
├── ml_eval.py        # ML-kNN fitness evaluator
├── mlknn_impl.py     # ML-kNN (PyTorch and scikit-learn backends)
├── nsga2.py          # NSGA-II engine
└── …                 # utilities
configs/              # hgemo.yaml (default), demo.yaml (quick check)
scripts/              # create_venv.sh, check_bundle.sh
data/Emotions/        # example dataset (5 folds)
```

## License

MIT License.

## Citation

If you use this code, please cite:

```bibtex
@article{hgemo2026,
  title   = {H-GEMO: Template-Based Hypergraph Genotypes for
             Multi-Objective Multi-Label Feature Selection},
  year    = {2026}
}
```
