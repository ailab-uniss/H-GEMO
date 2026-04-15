"""Experiment runner — the entry point for a single fold.

``run_experiment_from_config(config, fold_idx)`` performs:

1. Load pre-folded dataset (train / val / test).
2. Build the ML-kNN evaluator with tri-objective scoring.
3. Construct the hypergraph template (or bitstring config).
4. Run NSGA-II with sliding-window early stopping on HV.
5. Save Pareto front, population masks, and test-set metrics.

The function is called by ``hgemo.cli.run`` for each fold.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .config import Paths, get, load_config
from .datasets import load_dataset
from .genotypes import (
    BitstringConfig,
    HypergraphConfig,
    HypergraphGenome,
    bitstring_crossover,
    bitstring_mutate,
    build_feat_edge_membership,
    build_feature_similarity,
    clone_hypergraph,
    uniform_hyperedge_crossover,
    hyperedge_composite_mutation,
    repair_hypergraph,
    hypergraph_to_feature_mask,
    init_bitstring,
    init_hypergraph,
    build_hyperedges_multilayer,
)
from .metrics import hypervolume_3d, pareto_nondominated, pareto_nondominated_mask
from .ml_eval import EvalConfig, Evaluator
from .nsga2 import Variation, nsga2
from .utils import JsonlLogger
from .logging_utils import setup_run_logger


# ═══════════════════════════════════════════════════════════════════
# Null-distribution p95 for cosine similarity in R^L
# (precomputed via Monte Carlo, n = 200 k pairs).
# Used by the adaptive ``graph_contraction_threshold="auto"`` logic.
# ═══════════════════════════════════════════════════════════════════

_P95_NULL: list[tuple[int, float]] = [
    (2, 0.998), (3, 0.980), (4, 0.955), (5, 0.935),
    (6, 0.909), (7, 0.889), (8, 0.868), (9, 0.851),
    (10, 0.835), (12, 0.807), (14, 0.785), (16, 0.768),
    (18, 0.753), (20, 0.742), (22, 0.730), (25, 0.717),
    (30, 0.696), (35, 0.682), (40, 0.671), (45, 0.660),
    (50, 0.653), (60, 0.640), (80, 0.624), (100, 0.613),
]


def _p95_null_cosine(n_labels: int) -> float:
    """Return the 95th-percentile cosine similarity between random
    non-negative vectors in R^L (linear interpolation over table)."""
    if n_labels <= 1:
        return 1.0
    tbl = _P95_NULL
    if n_labels <= tbl[0][0]:
        return tbl[0][1]
    if n_labels >= tbl[-1][0]:
        L0, p0 = tbl[-2]; L1, p1 = tbl[-1]
        return max(p1 + (p1 - p0) / (L1 - L0) * (n_labels - L1), 0.50)
    for i in range(len(tbl) - 1):
        La, pa = tbl[i]; Lb, pb = tbl[i + 1]
        if La <= n_labels <= Lb:
            return pa + (pb - pa) * (n_labels - La) / (Lb - La)
    return 0.85


# ═══════════════════════════════════════════════════════════════════
# Variation adapters (bridge genotype operators → NSGA-II protocol)
# ═══════════════════════════════════════════════════════════════════

class BitstringVariation(Variation):
    """Variation operator for the flat bitstring genotype."""
    def __init__(self, cfg: BitstringConfig) -> None:
        self.cfg = cfg

    def crossover(self, a: object, b: object, rng: np.random.Generator) -> tuple[object, object]:
        return bitstring_crossover(np.asarray(a, dtype=bool), np.asarray(b, dtype=bool), rng)

    def mutate(self, a: object, rng: np.random.Generator) -> object:
        return bitstring_mutate(np.asarray(a, dtype=bool), self.cfg, rng)


class HypergraphVariation(Variation):
    """Variation operator for the hypergraph genotype.

    Wraps ``uniform_hyperedge_crossover``, ``hyperedge_composite_mutation``,
    and ``repair_hypergraph``.
    """
    def __init__(
        self,
        edges_template: list[Any],
        x_train: sparse.csr_matrix,
        cfg: HypergraphConfig,
    ) -> None:
        self.edges_template = edges_template
        self.x_train = x_train
        self.cfg = cfg
        self._feat_similarity = build_feature_similarity(edges_template, n_features=x_train.shape[1])
        self._feat_edge_membership = build_feat_edge_membership(edges_template, n_features=x_train.shape[1])
        try:
            nnz = np.asarray(x_train.getnnz(axis=0)).ravel().astype(np.float32)
            self._feature_freq = nnz / float(max(1, x_train.shape[0]))
        except Exception:
            self._feature_freq = None

    def crossover(self, a: object, b: object, rng: np.random.Generator) -> tuple[object, object]:
        return uniform_hyperedge_crossover(a, b, rng, swap_prob=0.5)

    def mutate(self, a: object, rng: np.random.Generator) -> object:
        return hyperedge_composite_mutation(
            a, self.edges_template, self.x_train, self.cfg, rng,
            feature_freq=self._feature_freq,
            feat_edge_membership=self._feat_edge_membership,
            feat_similarity=self._feat_similarity,
        )

    def repair(self, a: object, rng: np.random.Generator) -> object:
        return repair_hypergraph(
            a, self.edges_template, self.x_train, self.cfg, rng,
            feat_similarity=self._feat_similarity,
        )


# ═══════════════════════════════════════════════════════════════════
# Auto init_edge_prob
# ═══════════════════════════════════════════════════════════════════

def _resolve_init_edge_prob(raw: Any, n_features: int) -> float:
    """``"auto"`` → 0.3 + 0.3·exp(−D/50), otherwise parse as float."""
    import math
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        p = 0.3 + 0.3 * math.exp(-n_features / 50)
        logging.getLogger(__name__).info(
            "init_edge_prob=auto → %.4f  (D=%d)", p, n_features)
        return p
    return float(raw)


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════

def run_experiment_from_config(config: dict[str, Any], fold_idx: int | None = None) -> Path:
    """Run a single experiment from a configuration dictionary.

    Args:
        config: Parsed YAML configuration.
        fold_idx: Fold index for cross-validation (0-based).

    Returns:
        Path to the output directory.
    """
    seed = int(config.get("seed", 42))
    paths = Paths.from_config(config)
    if fold_idx is not None:
        suffix = f"_fold{int(fold_idx)}"
        if not paths.out_dir.name.endswith(suffix):
            paths = Paths(out_dir=paths.out_dir.with_name(paths.out_dir.name + suffix))
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    runlog = setup_run_logger(paths.out_dir, name=f"hgemo.run.{paths.out_dir.as_posix()}")
    log = runlog.logger
    logger = JsonlLogger(paths.out_dir / "history.jsonl")

    # Save the exact config used.
    try:
        import yaml
        with open(paths.out_dir / "config_used.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    except Exception:
        pass

    log.info("run.start out_dir=%s seed=%s fold_idx=%s", paths.out_dir, seed, fold_idx)

    # ── Load dataset ──────────────────────────────────────────────
    ds = load_dataset(config, seed=seed, fold_idx=fold_idx)
    n_features = ds.x_train.shape[1]
    n_labels = ds.y_train.shape[1]
    log.info("dataset x_train=%s y_train=%s x_val=%s y_val=%s",
             ds.x_train.shape, ds.y_train.shape, ds.x_val.shape, ds.y_val.shape)

    # ── Build evaluator ───────────────────────────────────────────
    obj_names = config.get("objectives", {}).get("names")
    if obj_names and isinstance(obj_names, list):
        objective_names = [Evaluator._canonical(str(n)) for n in obj_names]
    else:
        objective_names = None

    eval_cfg = EvalConfig(
        kind=str(get(config, "model.kind", "mlknn")),
        primary_objective=objective_names[0] if objective_names else "one_minus_macro_f1",
        objective_names=objective_names,
        random_state=seed,
        k=int(get(config, "model.k", 5)),
        s=float(get(config, "model.s", 1.0)),
        mlknn_backend=str(get(config, "model.mlknn_backend", "auto")),
        mlknn_device=str(get(config, "model.mlknn_device", "auto")),
    )
    evaluator = Evaluator(ds.x_train, ds.y_train, ds.x_val, ds.y_val, eval_cfg)
    log.info("evaluator kind=%s objectives=%s", eval_cfg.kind, objective_names)

    # ── Evolution parameters ──────────────────────────────────────
    evo_cfg = config.get("evolution", {})
    pop_size = int(evo_cfg.get("pop_size", 50))
    crossover_prob = float(evo_cfg.get("crossover_prob", 0.9))
    mutation_prob = float(evo_cfg.get("mutation_prob", 0.5))
    genotype_kind = str(evo_cfg.get("genotype", "hypergraph"))
    log.info("evolution genotype=%s pop=%d cx=%.2f mut=%.2f",
             genotype_kind, pop_size, crossover_prob, mutation_prob)

    counts = {"real": 0}

    # ── Genotype setup ────────────────────────────────────────────
    if genotype_kind == "bitstring":
        bcfg = BitstringConfig(
            init_prob=float(get(config, "bitstring.init_prob", 0.1)),
            bitflip_prob=float(get(config, "bitstring.bitflip_prob", 1.0 / max(1, n_features))),
            bitflip_prob_on=get(config, "bitstring.bitflip_prob_on", None),
            bitflip_prob_off=get(config, "bitstring.bitflip_prob_off", None),
        )
        variation: Variation = BitstringVariation(bcfg)

        def init_pop(rng: np.random.Generator) -> list[object]:
            return [init_bitstring(n_features, bcfg, rng) for _ in range(pop_size)]

        def to_mask(genome: object) -> np.ndarray:
            m = np.asarray(genome, dtype=bool)
            if m.sum() == 0:
                m = m.copy(); m[0] = True
            return m

    elif genotype_kind == "hypergraph":
        hcfg = HypergraphConfig(
            construction=str(get(config, "hypergraph.construction", "multilayer_label")),
            fl_relevance=str(get(config, "hypergraph.fl_relevance", "mi")),
            fl_mi_estimator=str(get(config, "hypergraph.fl_mi_estimator", "auto")),
            fl_mi_n_neighbors=int(get(config, "hypergraph.fl_mi_n_neighbors", 3)),
            fl_mi_cache=bool(get(config, "hypergraph.fl_mi_cache", True)),
            fl_topk_labels_per_feature=int(get(config, "hypergraph.fl_topk_labels_per_feature", 8)),
            fl_topm_ratio=float(get(config, "hypergraph.fl_topm_ratio", 0.15)),
            fl_topm_min=int(get(config, "hypergraph.fl_topm_min", 3)),
            ll_similarity=str(get(config, "hypergraph.ll_similarity", "jaccard")),
            ll_topk=int(get(config, "hypergraph.ll_topk", 30)),
            ll_min_cooccurrence=int(get(config, "hypergraph.ll_min_cooccurrence", 1)),
            topk_per_label=int(get(config, "hypergraph.topk_per_label", 30)),
            init_edge_prob=_resolve_init_edge_prob(
                get(config, "hypergraph.init_edge_prob", "auto"), n_features),
            graph_contraction_threshold=get(config, "hypergraph.graph_contraction_threshold", 0.7),
            edge_prune_prob=float(get(config, "hypergraph.edge_prune_prob", 0.05)),
            target_feature_ratio=get(config, "hypergraph.target_feature_ratio", None),
            min_feature_ratio=get(config, "hypergraph.min_feature_ratio", None),
            injection_prob=float(get(config, "hypergraph.injection_prob", 0.0)),
            injection_k=int(get(config, "hypergraph.injection_k", 1)),
            template_max_edges=get(config, "hypergraph.template_max_edges", None),
            feature_cluster_edges=get(config, "hypergraph.feature_cluster_edges", "auto"),
            mutation_routing_swap=float(get(config, "hypergraph.mutation_routing_swap", 0.40)),
            mutation_routing_inject=float(get(config, "hypergraph.mutation_routing_inject", 0.20)),
            edge_toggle_prob=float(get(config, "hypergraph.edge_toggle_prob", 0.50)),
        )
        cache_dir = str(get(config, "paths.cache_dir", "data/cache"))
        ds_name = str(get(config, "dataset.name", "")) or None
        edges_template, label_communities, _ = build_hyperedges_multilayer(
            ds.x_train, ds.y_train, hcfg,
            seed=seed, cache_dir=cache_dir, dataset_name=ds_name, fold_idx=fold_idx,
        )
        variation = HypergraphVariation(edges_template, ds.x_train, hcfg)

        # Resolve adaptive graph_contraction_threshold.
        _raw_gct = hcfg.graph_contraction_threshold
        if isinstance(_raw_gct, str) and _raw_gct.strip().lower() == "auto":
            _gct = max(_p95_null_cosine(n_labels), 0.85)
            log.info("graph_contraction_threshold=auto → %.4f (L=%d)", _gct, n_labels)
            hcfg = HypergraphConfig(**{**hcfg.__dict__, "graph_contraction_threshold": _gct})
            variation.cfg = hcfg
        elif isinstance(_raw_gct, str):
            hcfg = HypergraphConfig(**{**hcfg.__dict__, "graph_contraction_threshold": float(_raw_gct)})
            variation.cfg = hcfg

        log.info("hypergraph edges=%d comms=%s init_edge_prob=%.3f gct=%.3f",
                 len(edges_template),
                 len(label_communities) if label_communities else 0,
                 hcfg.init_edge_prob,
                 hcfg.graph_contraction_threshold if isinstance(hcfg.graph_contraction_threshold, (int, float)) else 0.7)

        def init_pop(rng: np.random.Generator) -> list[object]:
            return [init_hypergraph(edges_template, hcfg, rng) for _ in range(pop_size)]

        def to_mask(genome: object) -> np.ndarray:
            return hypergraph_to_feature_mask(genome, n_features=n_features)
    else:
        raise ValueError(f"Unsupported genotype: {genotype_kind!r} (bitstring | hypergraph)")

    # ── Evaluate function ─────────────────────────────────────────
    def evaluate(genome: object) -> tuple[np.ndarray, dict[str, object]]:
        if genotype_kind == "hypergraph":
            genome = clone_hypergraph(genome)
        mask = to_mask(genome)
        obj, ml = evaluator.evaluate_mask(mask)
        counts["real"] += 1
        return obj, {"selected": int(mask.sum()),
                      "feature_ratio": float(mask.sum() / mask.size),
                      "ml": asdict(ml)}

    # ── Early stopping (sliding-window on HV) ─────────────────────
    n_obj = len(objective_names) if objective_names else 2
    ref_list = [float(get(config, f"objectives.hv_ref.{i}", 1.0)) for i in range(n_obj)]
    ref = tuple(ref_list)

    es_cfg = evo_cfg.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", True))
    es_window = int(es_cfg.get("window", 10))
    es_rel_tol = float(es_cfg.get("rel_tol", 0.002))
    es_patience = int(es_cfg.get("patience", 2))

    hv_history: list[float] = []
    stagnant_count = 0
    last_gen = -1

    def on_generation(gen: int, pop: list[Any]) -> bool | None:
        nonlocal last_gen, stagnant_count
        last_gen = gen

        objs = np.stack([p.objectives for p in pop], axis=0)
        nd = pareto_nondominated(objs)
        hv = hypervolume_3d(nd, ref=ref) if n_obj == 3 else float(np.prod(np.array(ref[:2]) - nd.min(axis=0)))

        stop = False
        if es_enabled:
            hv_history.append(hv)
            if len(hv_history) >= 2 * es_window:
                prev_w = np.mean(hv_history[-(2 * es_window):-es_window])
                curr_w = np.mean(hv_history[-es_window:])
                rel = (curr_w - prev_w) / max(abs(prev_w), 1e-12)
                if rel < es_rel_tol:
                    stagnant_count += 1
                else:
                    stagnant_count = 0
                if stagnant_count >= es_patience:
                    log.info("Early stopping at gen=%d (HV stagnant for %d checks)", gen, es_patience)
                    stop = True

        log.info("gen=%d hv=%.6f nd=%d evals=%d wait=%d/%d",
                 gen, hv, nd.shape[0], counts["real"], stagnant_count, es_patience)
        logger.log({"gen": gen, "hv": hv, "n_nd": nd.shape[0], "evals": counts["real"],
                     "stagnant": stagnant_count})
        return stop

    # ── Budget ────────────────────────────────────────────────────
    max_evals_pf = evo_cfg.get("max_evals_per_feature")
    if max_evals_pf is not None:
        max_evals = int(n_features * float(max_evals_pf))
    else:
        max_evals = int(evo_cfg.get("max_evals", n_features * 100))
    log.info("max_evals=%d", max_evals)

    # ── Run NSGA-II ───────────────────────────────────────────────
    final_pop = nsga2(
        init_population=init_pop,
        evaluate=evaluate,
        variation=variation,
        pop_size=pop_size,
        max_evals=max_evals,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        seed=seed,
        on_generation=on_generation,
    )

    # ═══════════════════════════════════════════════════════════════
    # Post-processing: save results
    # ═══════════════════════════════════════════════════════════════

    # Pareto front from surrogate objectives.
    objs = np.stack([p.objectives for p in final_pop], axis=0)
    nd = pareto_nondominated(objs)
    np.savetxt(paths.out_dir / "pareto_front.csv", nd, delimiter=",")

    # Real-evaluated Pareto front (validation set).
    masks: list[np.ndarray] = []
    val_objs_list: list[np.ndarray] = []
    val_mls: list[Any] = []
    for ind in final_pop:
        g = clone_hypergraph(ind.genome) if genotype_kind == "hypergraph" else ind.genome
        mask = to_mask(g)
        vo, vm = evaluator.evaluate_mask(mask)
        masks.append(mask)
        val_objs_list.append(vo)
        val_mls.append(vm)

    val_objs_arr = np.stack(val_objs_list) if val_objs_list else np.zeros((0, n_obj))
    nd_val = pareto_nondominated(val_objs_arr)
    np.savetxt(paths.out_dir / "pareto_front_real.csv", nd_val, delimiter=",")

    # Test-set evaluation (train = train ∪ val).
    x_full = sparse.vstack([ds.x_train, ds.x_val]).tocsr()
    y_full = sparse.vstack([ds.y_train, ds.y_val]).tocsr()
    test_eval = Evaluator(x_full, y_full, ds.x_test, ds.y_test, eval_cfg)

    test_objs_list: list[np.ndarray] = []
    test_mls: list[Any] = []
    for mask in masks:
        to, tm = test_eval.evaluate_mask(mask)
        test_objs_list.append(to)
        test_mls.append(tm)
    test_objs_arr = np.stack(test_objs_list) if test_objs_list else np.zeros((0, n_obj))
    nd_test = pareto_nondominated(test_objs_arr)
    np.savetxt(paths.out_dir / "pareto_front_test.csv", nd_test, delimiter=",")

    # Population masks (packed, for post-hoc analysis).
    try:
        masks_arr = np.stack(masks).astype(np.uint8)
        np.savez_compressed(
            paths.out_dir / "population_masks.npz",
            masks_packed=np.packbits(masks_arr, axis=1),
            n_features=n_features,
            val_objs=val_objs_arr.astype(np.float32),
            test_objs=test_objs_arr.astype(np.float32),
            pareto_val_mask=pareto_nondominated_mask(val_objs_arr).astype(np.uint8),
        )
    except Exception:
        pass

    # Human-readable Pareto solutions (JSON).
    nd_mask = pareto_nondominated_mask(val_objs_arr)
    pareto_solutions = []
    for i, keep in enumerate(nd_mask.tolist()):
        if not bool(keep):
            continue
        m = masks[i]
        pareto_solutions.append({
            "i": i,
            "objectives": val_objs_arr[i].tolist(),
            "selected": int(m.sum()),
            "selected_features": np.flatnonzero(m).astype(int).tolist(),
            "val": asdict(val_mls[i]),
            "test": asdict(test_mls[i]),
        })
    (paths.out_dir / "pareto_val_solutions.json").write_text(json.dumps({
        "train_for_test": "train+val",
        "model_kind": eval_cfg.kind,
        "fold_idx": fold_idx,
        "n_features": n_features,
        "solutions": pareto_solutions,
    }, indent=2))

    # Budget-selected solutions (select by val, report test).
    budget_mode = str(get(config, "reporting.budget_mode", "count"))
    budget_step = int(get(config, "reporting.budget_step", 1))
    max_ratio = float(get(config, "reporting.max_feature_ratio", 0.25))
    M = int(np.floor(max_ratio * n_features))
    budgets = list(range(1, max(1, M) + 1, budget_step))

    feat_col = int(objective_names.index("feature_ratio")) if (objective_names and "feature_ratio" in objective_names) else 1
    selected_counts = np.array([int(m.sum()) for m in masks], dtype=int)
    by_budget: list[dict] = []

    for budget in budgets:
        eligible = np.flatnonzero(selected_counts <= int(budget))
        if eligible.size == 0:
            continue
        best_i = int(min(eligible.tolist(),
                         key=lambda i: (float(val_objs_arr[i, 0]), float(val_objs_arr[i, feat_col]))))
        by_budget.append({
            "budget": int(budget),
            "selected": int(masks[best_i].sum()),
            "selected_features": np.flatnonzero(masks[best_i]).astype(int).tolist(),
            "val": asdict(val_mls[best_i]),
            "test": asdict(test_mls[best_i]),
        })
    (paths.out_dir / "test_selected_by_val.json").write_text(json.dumps({
        "train_for_test": "train+val",
        "model_kind": eval_cfg.kind,
        "fold_idx": fold_idx,
        "budget_mode": budget_mode,
        "by_budget": by_budget,
    }, indent=2))

    # Summary.
    hv_val = hypervolume_3d(nd_val, ref=ref) if n_obj == 3 else 0.0
    hv_test = hypervolume_3d(nd_test, ref=ref) if n_obj == 3 else 0.0
    (paths.out_dir / "summary.json").write_text(json.dumps({
        "seed": seed, "fold_idx": fold_idx,
        "dataset": config.get("dataset", {}),
        "evolution": config.get("evolution", {}),
        "final": {
            "gen": last_gen, "pareto_points_val": nd_val.shape[0],
            "hv_val": hv_val, "pareto_points_test": nd_test.shape[0],
            "hv_test": hv_test, "total_evals": counts["real"],
        },
    }, indent=2))

    log.info("run.done out_dir=%s pareto=%d", paths.out_dir, nd_val.shape[0])
    runlog.close()
    return paths.out_dir


def run_experiment(config_path: str | Path) -> Path:
    """Load YAML config and run."""
    return run_experiment_from_config(load_config(config_path))
