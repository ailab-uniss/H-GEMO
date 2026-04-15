"""Genotype representations used by H-GEMO.

This module implements the two genotypes evaluated in the paper:

1. **Bitstring** — Standard binary mask (used for the ablation study).
2. **Hypergraph** — Multilayer-label hypergraph construction with
   MI-based feature–label relevance, Jaccard label–label similarity,
   KMeans feature-cluster edges, and composite mutation.

Only the ``multilayer_label`` construction mode is included (the only
mode used in the published experiments).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.feature_selection import mutual_info_classif
import warnings
from joblib import Parallel, delayed


# ═══════════════════════════════════════════════════════════════════
# §1  Configuration dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BitstringConfig:
    """Parameters for the bitstring genotype (ablation baseline)."""
    init_prob: float = 0.1            # probability each feature is ON at init
    bitflip_prob: float = 0.01        # symmetric per-bit flip probability
    bitflip_prob_on: float | None = None   # asymmetric 1→0 flip (overrides bitflip_prob)
    bitflip_prob_off: float | None = None  # asymmetric 0→1 flip (overrides bitflip_prob)


@dataclass(frozen=True)
class HypergraphConfig:
    """Parameters for the hypergraph genotype (paper default).

    The published experiments use ``construction="multilayer_label"``
    with MI-based feature–label relevance and Jaccard label–label
    similarity.  Feature-cluster edges (KMeans on MI profiles) are
    added when ``feature_cluster_edges`` is set to ``"auto"`` or an
    explicit integer.
    """
    # ── Hyperedge construction ────────────────────────────────────
    construction: str = "multilayer_label"

    # ── Feature–label relevance (MI) ──────────────────────────────
    fl_relevance: str = "mi"
    fl_mi_estimator: str = "auto"     # auto | binary_discrete | sklearn_knn | sklearn_discrete
    fl_mi_n_neighbors: int = 3        # k for sklearn_knn estimator
    fl_mi_cache: bool = True
    fl_topk_labels_per_feature: int = 8
    fl_topm_ratio: float = 0.15       # fraction of D kept per label edge
    fl_topm_min: int = 3              # minimum features per label edge

    # ── Label–label similarity ────────────────────────────────────
    ll_similarity: str = "jaccard"    # jaccard | cosine
    ll_topk: int = 30
    ll_min_cooccurrence: int = 1

    # ── Hypergraph init / structure ───────────────────────────────
    topk_per_label: int = 30
    init_edge_prob: float = 0.3       # can also be "auto"
    graph_contraction_threshold: float | str = 0.7   # or "auto"
    edge_prune_prob: float = 0.05
    target_feature_ratio: float | None = None
    min_feature_ratio: float | None = None
    injection_prob: float = 0.0
    injection_k: int = 1
    template_max_edges: int | None = None

    # ── Feature-cluster edges (KMeans on MI profiles) ─────────────
    feature_cluster_edges: str | int | None = "auto"

    # ── Mutation routing probabilities ────────────────────────────
    mutation_routing_swap: float = 0.40     # count-neutral swap
    mutation_routing_inject: float = 0.30   # constructive injection
    # skip = 1 - swap - inject (derived)
    edge_toggle_prob: float = 0.50          # probability of toggling one edge


# ═══════════════════════════════════════════════════════════════════
# §2  Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Hyperedge:
    """A single hyperedge in the template graph.

    Each hyperedge links a set of *labels* to the *features* most
    relevant to them, scored by MI (or aggregated MI for cluster edges).
    """
    labels: np.ndarray     # 1-D int label indices
    features: np.ndarray   # 1-D int feature indices
    scores: np.ndarray | None = None   # per-feature MI score (higher ↔ better)


@dataclass
class HypergraphGenome:
    """Individual genome: a subset of template hyperedges with per-edge feature lists."""
    active_edges: np.ndarray          # bool [n_edges]
    edge_features: list[np.ndarray]   # per-edge feature list (may shrink during evolution)


# ═══════════════════════════════════════════════════════════════════
# §3  Bitstring operators
# ═══════════════════════════════════════════════════════════════════

def init_bitstring(n_features: int, cfg: BitstringConfig, rng: np.random.Generator) -> np.ndarray:
    """Create a random binary mask with P(bit=1) = init_prob."""
    mask = rng.random(n_features) < float(cfg.init_prob)
    if mask.sum() == 0:
        mask[rng.integers(0, n_features)] = True
    return mask


def bitstring_crossover(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover: each bit is independently taken from either parent."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    m = rng.random(a.size) < 0.5
    c1, c2 = np.where(m, a, b), np.where(m, b, a)
    for c in (c1, c2):
        if c.sum() == 0:
            cands = np.flatnonzero(a | b)
            idx = int(rng.choice(cands)) if cands.size else int(rng.integers(0, a.size))
            c[idx] = True
    return c1, c2


def bitstring_mutate(a: np.ndarray, cfg: BitstringConfig, rng: np.random.Generator) -> np.ndarray:
    """Bit-flip mutation with optional asymmetric probabilities."""
    a = np.asarray(a, dtype=bool).copy()
    p_on = float(cfg.bitflip_prob_on if cfg.bitflip_prob_on is not None else cfg.bitflip_prob)
    p_off = float(cfg.bitflip_prob_off if cfg.bitflip_prob_off is not None else cfg.bitflip_prob)
    r = rng.random(a.size)
    flips = (a & (r < p_on)) | ((~a) & (r < p_off))
    a ^= flips.astype(bool, copy=False)
    if a.sum() == 0:
        a[rng.integers(0, a.size)] = True
    return a


# ═══════════════════════════════════════════════════════════════════
# §4  Feature–label mutual information
# ═══════════════════════════════════════════════════════════════════

def compute_feature_label_mi(
    x: sparse.csr_matrix,
    y: sparse.csr_matrix,
    *,
    seed: int,
    n_neighbors: int = 3,
    estimator: str = "auto",
    discrete_features: bool | str = "auto",
    n_jobs: int = -1,
) -> np.ndarray:
    """Compute feature–label MI matrix ``[n_features, n_labels]``.

    Three estimators are available:

    * ``binary_discrete`` — fast custom MI on binarised X (presence/absence).
    * ``sklearn_discrete`` — ``mutual_info_classif`` with ``discrete_features=True``.
    * ``sklearn_knn`` — ``mutual_info_classif`` with kNN estimator.

    ``"auto"`` selects ``binary_discrete`` for sparse binary-like data,
    ``sklearn_discrete`` for high-dimensional sparse data, and
    ``sklearn_knn`` otherwise.
    """
    x = x.tocsr()
    y = y.tocsr()

    # ── Auto-selection heuristic ──────────────────────────────────

    def _looks_binary_sparse(x_csr: sparse.csr_matrix) -> bool:
        d = x_csr.data
        if d.size == 0:
            return True
        if float(d.min()) < 0.0 or float(d.max()) > 1.0:
            return False
        return bool(np.allclose(d[: min(50_000, d.size)], 1.0))

    def _resolve_estimator(x_csr: sparse.csr_matrix, est_in: str) -> str:
        est = str(est_in).strip().lower()
        if est != "auto":
            return est
        if sparse.issparse(x_csr):
            n_s, n_f = x_csr.shape
            if n_f > 0 and n_s > 0:
                col_nnz = np.asarray(x_csr.getnnz(axis=0)).ravel().astype(np.float32)
                frac_nz = col_nnz / float(max(1, n_s))
                if float(np.mean(frac_nz < 0.5)) >= 0.25:
                    return "binary_discrete"
            if x_csr.shape[1] >= 2000:
                return "binary_discrete" if _looks_binary_sparse(x_csr) else "sklearn_discrete"
        return "sklearn_knn"

    # ── Binary-discrete MI (fast, sparse-friendly) ────────────────

    def _mi_binary_discrete(x_csr: sparse.csr_matrix, y_csr: sparse.csr_matrix) -> np.ndarray:
        n = int(x_csr.shape[0])
        n_features = int(x_csr.shape[1])
        y_dense = y_csr.toarray().astype(np.int8, copy=False)
        n_labels = int(y_dense.shape[1])
        col_nnz = np.asarray(x_csr.getnnz(axis=0)).ravel().astype(np.int64)

        def _mi_one_label(l: int) -> np.ndarray:
            y_l = y_dense[:, l].astype(bool)
            pos_idx = np.flatnonzero(y_l)
            n_pos = pos_idx.size
            if n_pos == 0 or n_pos == n:
                return np.zeros(n_features, dtype=np.float32)
            n11 = np.asarray(x_csr[pos_idx].getnnz(axis=0)).ravel().astype(np.int64)
            n10 = col_nnz - n11
            n01 = n_pos - n11
            n00 = n - n11 - n10 - n01
            n_f = float(n)
            p_x1 = col_nnz / n_f
            p_x0 = 1.0 - p_x1
            p_y1 = float(n_pos) / n_f
            p_y0 = 1.0 - p_y1
            out = np.zeros(n_features, dtype=np.float64)
            for nxy, px, py in ((n11, p_x1, p_y1), (n10, p_x1, p_y0),
                                (n01, p_x0, p_y1), (n00, p_x0, p_y0)):
                mask = nxy > 0
                if not np.any(mask):
                    continue
                pxy = nxy[mask].astype(np.float64) / n_f
                denom = px[mask].astype(np.float64) * float(py)
                out[mask] += pxy * np.log(pxy / denom)
            return out.astype(np.float32)

        mi = np.zeros((n_features, n_labels), dtype=np.float32)
        for l in range(n_labels):
            mi[:, l] = _mi_one_label(l)
        return mi

    # ── Dispatch ──────────────────────────────────────────────────

    est = _resolve_estimator(x, str(estimator).strip().lower()) if str(estimator).strip().lower() == "auto" else str(estimator).strip().lower()

    if est == "binary_discrete":
        return _mi_binary_discrete(x, y)

    # sklearn-based estimators
    y_dense = y.toarray().astype(int, copy=False)
    n_labels = y_dense.shape[1]
    n_features = x.shape[1]
    if est == "sklearn_discrete":
        discrete_features = True
        n_neighbors = 3

    def _mi_one_label_sklearn(l: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return mutual_info_classif(
                x, y_dense[:, l],
                n_neighbors=int(n_neighbors),
                discrete_features=discrete_features,
                random_state=seed,
            ).astype(np.float32)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_mi_one_label_sklearn)(l) for l in range(n_labels)
    )
    return np.column_stack(results) if n_labels > 0 else np.zeros((n_features, 0), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# §5  MI cache
# ═══════════════════════════════════════════════════════════════════

def _mi_cache_key(
    x: sparse.csr_matrix,
    y: sparse.csr_matrix,
    *,
    estimator: str,
    seed: int,
    n_neighbors: int,
    fold_idx: int | None,
) -> str:
    """SHA-1 fingerprint of the MI computation inputs (for disk caching)."""
    import hashlib
    x, y = x.tocsr(), y.tocsr()
    h = hashlib.sha1()
    h.update(str((x.shape[0], x.shape[1], x.nnz)).encode())
    h.update(str((y.shape[0], y.shape[1], y.nnz)).encode())
    h.update(str((str(estimator), int(seed), int(n_neighbors), fold_idx)).encode())
    h.update(np.asarray(x.indptr[: min(10_000, x.indptr.size)], dtype=np.int64).tobytes())
    h.update(np.asarray(x.indices[: min(200_000, x.indices.size)], dtype=np.int32).tobytes())
    h.update(np.asarray(y.indptr[: min(10_000, y.indptr.size)], dtype=np.int64).tobytes())
    h.update(np.asarray(y.indices[: min(200_000, y.indices.size)], dtype=np.int32).tobytes())
    h.update(np.asarray(y.sum(axis=0)).ravel().astype(np.int64).tobytes())
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════
# §6  Helper array utilities
# ═══════════════════════════════════════════════════════════════════

def _topk_per_row(values: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-*k* values per row (2-D array)."""
    k = min(int(k), values.shape[1])
    idx = np.argpartition(values, -k, axis=1)[:, -k:]
    row_scores = np.take_along_axis(values, idx, axis=1)
    order = np.argsort(-row_scores, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def _topk_1d(values: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-*k* values in a 1-D array."""
    k = min(int(k), values.size)
    idx = np.argpartition(values, -k)[-k:]
    return idx[np.argsort(-values[idx])]


# ═══════════════════════════════════════════════════════════════════
# §7  Label–label similarity graph & community detection
# ═══════════════════════════════════════════════════════════════════

def build_label_label_similarity_graph(
    y: sparse.csr_matrix,
    *,
    topk: int,
    min_cooccurrence: int,
    similarity: str = "jaccard",
) -> tuple[nx.Graph, list[set[int]]]:
    """Build a weighted label–label graph and detect communities.

    Similarity measures:
    * ``jaccard``:  J(L_i,L_j) = |L_i∩L_j| / |L_i∪L_j|
    * ``cosine``:   cos(L_i,L_j) = |L_i∩L_j| / sqrt(|L_i|·|L_j|)

    Communities are detected with Louvain (fallback: greedy modularity)
    and used for community-preserving crossover.
    """
    y = y.tocsr()
    n_labels = y.shape[1]
    counts = np.asarray(y.sum(axis=0)).ravel().astype(np.float64)
    sim = str(similarity).strip().lower()

    inter = (y.T @ y).tocoo()
    neigh: list[list[tuple[int, float]]] = [[] for _ in range(n_labels)]
    for i, j, v in zip(inter.row, inter.col, inter.data):
        i, j, v = int(i), int(j), float(v)
        if i == j or v < float(min_cooccurrence):
            continue
        if sim == "jaccard":
            denom = counts[i] + counts[j] - v
        else:
            denom = float(np.sqrt(counts[i] * counts[j]))
        if denom <= 0:
            continue
        w = v / denom
        if not np.isfinite(w) or w <= 0.0:
            continue
        neigh[i].append((j, w))

    g = nx.Graph()
    g.add_nodes_from(range(n_labels))
    for i in range(n_labels):
        if not neigh[i]:
            continue
        neigh[i].sort(key=lambda t: t[1], reverse=True)
        for j, w in neigh[i][: int(min(topk, len(neigh[i])))]:
            g.add_edge(i, j, weight=float(w))

    # Community detection (Louvain preferred).
    communities: list[set[int]] = []
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(g, seed=0, weight="weight")
        communities = [set(map(int, c)) for c in comms if len(c) > 0]
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = greedy_modularity_communities(g, weight="weight")
        communities = [set(map(int, c)) for c in comms if len(c) > 0]

    # Ensure every label is covered.
    covered = set().union(*communities) if communities else set()
    for l in range(n_labels):
        if l not in covered:
            communities.append({l})

    # Degenerate: collapse into singletons when single community or nearly all singletons.
    if len(communities) <= 1:
        communities = [{l} for l in range(n_labels)]
    else:
        n_sing = sum(1 for c in communities if len(c) <= 1)
        if (n_sing / max(1, len(communities))) >= 0.90:
            communities = [{l} for l in range(n_labels)]

    return g, communities


# ═══════════════════════════════════════════════════════════════════
# §8  Multilayer-label hyperedge construction
# ═══════════════════════════════════════════════════════════════════

def build_hyperedges_multilayer(
    x: sparse.csr_matrix,
    y: sparse.csr_matrix,
    cfg: HypergraphConfig,
    *,
    seed: int,
    cache_dir: str | None = None,
    dataset_name: str | None = None,
    fold_idx: int | None = None,
) -> tuple[list[Hyperedge], list[set[int]] | None, None]:
    """Build hyperedge template using multilayer-label construction.

    Steps:
    1. Compute feature–label MI (or load from cache).
    2. Sparsify via top-k labels per feature + top-m features per label.
    3. Build one hyperedge per label.
    4. (Optionally) add KMeans feature-cluster edges.
    5. Detect label communities for crossover.

    Returns ``(edges, label_communities, None)``.
    """
    x, y = x.tocsr(), y.tocsr()
    n_features = x.shape[1]
    n_labels = y.shape[1]

    # ── Step 1: Compute MI ────────────────────────────────────────
    mi: np.ndarray | None = None
    cache_path = None

    # Resolve estimator name for caching (avoid keying by "auto").
    est_in = str(cfg.fl_mi_estimator).strip().lower()
    est_resolved = est_in
    if est_in == "auto":
        col_nnz = np.asarray(x.getnnz(axis=0)).ravel().astype(np.float32)
        frac_nz = col_nnz / float(max(1, x.shape[0]))
        frac_sparse = float(np.mean(frac_nz < 0.5))
        if frac_sparse >= 0.25:
            est_resolved = "binary_discrete"
        elif x.shape[1] >= 2000:
            d = x.data
            looks_binary = (d.size == 0 or
                            (float(d.min()) >= 0.0 and float(d.max()) <= 1.0 and
                             bool(np.allclose(d[: min(50_000, d.size)], 1.0))))
            est_resolved = "binary_discrete" if looks_binary else "sklearn_discrete"
        else:
            est_resolved = "sklearn_knn"

    if cfg.fl_mi_cache and cache_dir:
        from pathlib import Path as _P
        base = _P(cache_dir) / "mi"
        if dataset_name:
            base = base / str(dataset_name)
        base.mkdir(parents=True, exist_ok=True)
        key = _mi_cache_key(x, y, estimator=est_resolved, seed=seed,
                            n_neighbors=cfg.fl_mi_n_neighbors, fold_idx=fold_idx)
        cache_path = base / f"mi_{key}.npz"
        if cache_path.exists():
            try:
                mi = np.load(cache_path, allow_pickle=False)["mi"].astype(np.float32)
            except Exception:
                mi = None

    if mi is None:
        mi = compute_feature_label_mi(x, y, seed=seed,
                                      n_neighbors=cfg.fl_mi_n_neighbors,
                                      estimator=est_resolved)
        if cache_path is not None:
            try:
                np.savez_compressed(cache_path, mi=mi.astype(np.float32))
            except Exception:
                pass

    # ── Step 2: Sparsify ──────────────────────────────────────────
    topk_labels = _topk_per_row(mi, k=cfg.fl_topk_labels_per_feature)
    topm = max(int(cfg.fl_topm_min), int(np.floor(cfg.fl_topm_ratio * n_features)))
    topm = min(topm, n_features)

    topm_feats_per_label: list[np.ndarray] = []
    for l in range(n_labels):
        topm_feats_per_label.append(_topk_1d(mi[:, l], k=topm).astype(int))

    features_for_label: list[set[int]] = [
        set(map(int, topm_feats_per_label[l].tolist())) for l in range(n_labels)
    ]
    for f in range(n_features):
        for l in topk_labels[f]:
            features_for_label[int(l)].add(int(f))

    # ── Step 3: Label communities ─────────────────────────────────
    _, label_communities = build_label_label_similarity_graph(
        y, topk=cfg.ll_topk, min_cooccurrence=cfg.ll_min_cooccurrence,
        similarity=cfg.ll_similarity,
    )

    # ── Step 4: One hyperedge per label ───────────────────────────
    edges: list[Hyperedge] = []
    for l in range(n_labels):
        feats = np.fromiter(features_for_label[l], dtype=int)
        if feats.size == 0:
            feats = topm_feats_per_label[l][:1].copy()
        if feats.size > topm:
            scores_all = mi[feats, l]
            keep_idx = _topk_1d(scores_all, k=topm)
            feats = feats[keep_idx]
        scores = mi[feats, l].astype(np.float32)
        edges.append(Hyperedge(labels=np.array([l], dtype=int),
                               features=feats.astype(int), scores=scores))

    # ── Step 5: Feature-cluster edges (KMeans on MI profiles) ─────
    _fc_setting = getattr(cfg, "feature_cluster_edges", None)
    _fc_n = 0
    if _fc_setting is not None:
        if isinstance(_fc_setting, str) and _fc_setting.strip().lower() == "auto":
            _fc_n = max(2 * n_labels, int(np.ceil(n_features / max(1, topm))))
            _fc_n = min(_fc_n, n_features // 2)
        elif isinstance(_fc_setting, (int, float)) and int(_fc_setting) > 0:
            _fc_n = int(_fc_setting)
    if _fc_n > 1:
        from sklearn.cluster import KMeans
        mi_profiles = mi.copy()
        row_norms = np.linalg.norm(mi_profiles, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        mi_profiles = mi_profiles / row_norms
        _fc_n = min(_fc_n, n_features)
        km = KMeans(n_clusters=_fc_n, n_init=3, max_iter=50, random_state=seed)
        labels_km = km.fit_predict(mi_profiles)
        for c in range(_fc_n):
            feats_c = np.where(labels_km == c)[0].astype(int)
            if feats_c.size == 0:
                continue
            if feats_c.size > topm:
                total_mi = mi[feats_c].sum(axis=1)
                order = np.argsort(-total_mi)
                feats_c = feats_c[order]
                for start in range(0, feats_c.size, topm):
                    sub = feats_c[start : start + topm]
                    scores_sub = mi[sub].mean(axis=1).astype(np.float32)
                    cluster_mi = mi[sub].mean(axis=0)
                    assoc_labels = np.argsort(-cluster_mi)[:min(3, n_labels)].astype(int)
                    edges.append(Hyperedge(labels=assoc_labels, features=sub, scores=scores_sub))
            else:
                scores_c = mi[feats_c].mean(axis=1).astype(np.float32)
                cluster_mi = mi[feats_c].mean(axis=0)
                assoc_labels = np.argsort(-cluster_mi)[:min(3, n_labels)].astype(int)
                edges.append(Hyperedge(labels=assoc_labels, features=feats_c, scores=scores_c))

    # ── Step 6: Optional template cap ─────────────────────────────
    if cfg.template_max_edges is not None and len(edges) > int(cfg.template_max_edges):
        max_e = int(cfg.template_max_edges)
        scored = [(i, float(np.sum(e.scores)) if e.scores is not None else float(len(e.features)))
                  for i, e in enumerate(edges)]
        scored.sort(key=lambda t: (-t[1], t[0]))
        keep = sorted(i for i, _ in scored[:max_e])
        edges = [edges[i] for i in keep]

    return edges, label_communities, None


# ═══════════════════════════════════════════════════════════════════
# §9  Hypergraph genome helpers
# ═══════════════════════════════════════════════════════════════════

def init_hypergraph(
    edges: list[Hyperedge], cfg: HypergraphConfig, rng: np.random.Generator,
) -> HypergraphGenome:
    """Create a random hypergraph genome by activating edges with probability ``init_edge_prob``."""
    active = rng.random(len(edges)) < float(cfg.init_edge_prob)
    if active.sum() == 0:
        active[rng.integers(0, len(edges))] = True
    return HypergraphGenome(active_edges=active,
                            edge_features=[e.features.copy() for e in edges])


def hypergraph_to_feature_mask(genome: HypergraphGenome, n_features: int) -> np.ndarray:
    """Decode a hypergraph genome into a boolean feature mask."""
    mask = np.zeros(n_features, dtype=bool)
    for is_on, feats in zip(genome.active_edges, genome.edge_features):
        if bool(is_on):
            mask[np.asarray(feats, dtype=int)] = True
    if mask.sum() == 0:
        mask[0] = True
    return mask


def clone_hypergraph(g: HypergraphGenome) -> HypergraphGenome:
    """Deep copy of a hypergraph genome."""
    return HypergraphGenome(active_edges=g.active_edges.copy(),
                            edge_features=[f.copy() for f in g.edge_features])


# ═══════════════════════════════════════════════════════════════════
# §10  Feature–edge membership & similarity (for contraction)
# ═══════════════════════════════════════════════════════════════════

def build_feat_edge_membership(
    edges_template: list[Hyperedge], n_features: int,
) -> list[frozenset[int]]:
    """Map each feature to the set of template hyperedge indices it belongs to."""
    membership: list[set[int]] = [set() for _ in range(n_features)]
    for eidx, e in enumerate(edges_template):
        for f in np.asarray(e.features, dtype=int).tolist():
            if 0 <= int(f) < n_features:
                membership[int(f)].add(eidx)
    return [frozenset(s) for s in membership]


def build_feature_similarity(
    edges_template: list[Hyperedge], n_features: int,
) -> np.ndarray:
    """Build pairwise cosine similarity from template MI profiles.

    Features with similar MI profiles across labels are likely
    redundant.  Used by graph contraction to prune redundant features
    within a hyperedge.

    Returns ``[n_features, n_features]`` similarity matrix (diagonal=0).
    """
    n_labels = 0
    for e in edges_template:
        for l in np.asarray(e.labels, dtype=int).tolist():
            if int(l) + 1 > n_labels:
                n_labels = int(l) + 1
    if n_labels == 0:
        return np.zeros((n_features, n_features), dtype=np.float32)

    mi_profile = np.zeros((n_features, n_labels), dtype=np.float32)
    for e in edges_template:
        labels = np.asarray(e.labels, dtype=int)
        features = np.asarray(e.features, dtype=int)
        if e.scores is None:
            continue
        scores_arr = np.asarray(e.scores, dtype=np.float32)
        for li, l_val in enumerate(labels.tolist()):
            l_int = int(l_val)
            if l_int >= n_labels:
                continue
            for fi, f_val in enumerate(features.tolist()):
                f_int = int(f_val)
                if fi >= scores_arr.size or f_int >= n_features:
                    continue
                val = float(scores_arr[fi])
                if val > mi_profile[f_int, l_int]:
                    mi_profile[f_int, l_int] = val

    norms = np.linalg.norm(mi_profile, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    mi_normed = mi_profile / norms
    similarity = (mi_normed @ mi_normed.T).astype(np.float32)
    np.fill_diagonal(similarity, 0.0)
    return similarity


# ═══════════════════════════════════════════════════════════════════
# §11  Crossover
# ═══════════════════════════════════════════════════════════════════

def uniform_hyperedge_crossover(
    a: HypergraphGenome, b: HypergraphGenome, rng: np.random.Generator,
    *, swap_prob: float = 0.5,
) -> tuple[HypergraphGenome, HypergraphGenome]:
    """Uniform hyperedge crossover: swap each edge independently with P=swap_prob."""
    mask = rng.random(a.active_edges.size) < swap_prob
    c1 = HypergraphGenome(
        active_edges=np.where(mask, b.active_edges, a.active_edges).copy(),
        edge_features=[(fb if bool(m) else fa).copy()
                       for m, fa, fb in zip(mask, a.edge_features, b.edge_features)],
    )
    c2 = HypergraphGenome(
        active_edges=np.where(mask, a.active_edges, b.active_edges).copy(),
        edge_features=[(fa if bool(m) else fb).copy()
                       for m, fa, fb in zip(mask, a.edge_features, b.edge_features)],
    )
    for c in (c1, c2):
        if c.active_edges.sum() == 0:
            c.active_edges[int(rng.integers(0, len(c.active_edges)))] = True
    return c1, c2


# ═══════════════════════════════════════════════════════════════════
# §12  Composite mutation
# ═══════════════════════════════════════════════════════════════════

def hyperedge_composite_mutation(
    genome: HypergraphGenome,
    edges_template: list[Hyperedge],
    x_train: sparse.csr_matrix,
    cfg: HypergraphConfig,
    rng: np.random.Generator,
    feature_knn: list[np.ndarray] | None = None,
    feature_freq: np.ndarray | None = None,
    feature_label_skew: np.ndarray | None = None,
    feat_edge_membership: list[frozenset[int]] | None = None,
    feat_similarity: np.ndarray | None = None,
) -> HypergraphGenome:
    """Composite mutation operator (Algorithm 2 in the paper).

    Pipeline per call:

    1. **Edge prune** — disable one active edge with P = ``edge_prune_prob``.
    2. **Edge toggle** — flip one random edge with P = ``edge_toggle_prob``.
       When toggled ON, apply graph contraction.
    3. **Per-edge mutation** on one random active edge (routing):
       - *swap* (P = ``mutation_routing_swap``): replace one low-MI
         feature with a high-MI missing feature (count-neutral).
       - *inject* (P = ``mutation_routing_inject``): add features
         from the template (constructive).
       - *skip* (remainder): no per-edge change.
    4. **Graph contraction** — remove features whose cosine similarity
       to a retained feature exceeds ``graph_contraction_threshold``.
    5. **Sparsification / densification** — enforce
       ``target_feature_ratio`` / ``min_feature_ratio``.
    """
    g = HypergraphGenome(active_edges=genome.active_edges.copy(),
                         edge_features=[f.copy() for f in genome.edge_features])

    # Score map cache (per mutation call).
    score_maps: dict[int, dict[int, float]] = {}

    def _score_map_for_edge(edge_idx: int) -> dict[int, float] | None:
        if edge_idx in score_maps:
            return score_maps[edge_idx]
        scores = edges_template[edge_idx].scores
        if scores is None:
            return None
        feats = np.asarray(edges_template[edge_idx].features, dtype=int)
        if feats.size == 0:
            return None
        s = np.maximum(np.asarray(scores, dtype=np.float32), 0.0)
        if feature_freq is not None:
            try:
                freq = np.asarray(feature_freq, dtype=np.float32)[feats]
                floor = 1.0 / float(max(1, x_train.shape[0]))
                freq = np.maximum(freq, floor)
                s = s / freq
            except Exception:
                pass
        m = {int(f): float(s[i]) for i, f in enumerate(feats.tolist())}
        score_maps[edge_idx] = m
        return m

    # ── 1. Edge prune ─────────────────────────────────────────────
    if rng.random() < float(cfg.edge_prune_prob) and g.active_edges.sum() > 1:
        active_idxs = np.flatnonzero(g.active_edges)
        g.active_edges[int(rng.choice(active_idxs))] = False

    # ── 2. Edge toggle ────────────────────────────────────────────
    if rng.random() < float(cfg.edge_toggle_prob):
        idx = int(rng.integers(0, len(g.active_edges)))
        was_off = not bool(g.active_edges[idx])
        g.active_edges[idx] = ~g.active_edges[idx]
        if g.active_edges.sum() == 0:
            g.active_edges[idx] = True
        # When toggled ON → apply contraction to avoid bloat.
        if was_off and bool(g.active_edges[idx]) and feat_similarity is not None:
            thr = float(cfg.graph_contraction_threshold) if not isinstance(cfg.graph_contraction_threshold, str) else 0.7
            if thr < 1.0:
                feats_t = g.edge_features[idx]
                if feats_t.size > 1:
                    sm = _score_map_for_edge(idx)
                    if sm:
                        sf = sorted(feats_t.tolist(), key=lambda f: sm.get(int(f), 0.0), reverse=True)
                    else:
                        sf = list(feats_t)
                        rng.shuffle(sf)
                    keep = [sf[0]]
                    for f in sf[1:]:
                        if not any(feat_similarity[int(f), int(kf)] >= thr for kf in keep):
                            keep.append(f)
                    g.edge_features[idx] = np.array(sorted(set(keep)), dtype=int)
                    if g.edge_features[idx].size == 0:
                        g.edge_features[idx] = edges_template[idx].features[:1].copy()

    # ── 3. Per-edge mutation ──────────────────────────────────────
    active_idxs = np.flatnonzero(g.active_edges)
    if active_idxs.size == 0:
        return g

    eidx = int(rng.choice(active_idxs))
    scores = edges_template[eidx].scores
    score_map = _score_map_for_edge(eidx)
    if score_map is None and scores is not None:
        score_map = {int(f): float(scores[i])
                     for i, f in enumerate(edges_template[eidx].features.tolist())}

    roll = rng.random()
    swap_end = float(cfg.mutation_routing_swap)
    inject_end = swap_end + float(cfg.mutation_routing_inject)

    if roll < swap_end:
        # Feature swap (count-neutral).
        cur = g.edge_features[eidx].astype(int)
        tpl_feats = edges_template[eidx].features.astype(int)
        cur_set = set(map(int, cur.tolist()))
        missing = [int(f) for f in tpl_feats.tolist() if int(f) not in cur_set]
        if cur.size > 1 and missing and score_map:
            cur_scores = np.array([score_map.get(int(f), 0.0) for f in cur.tolist()], dtype=float)
            inv = 1.0 / np.maximum(cur_scores, 1e-12)
            inv /= inv.sum()
            rem_idx = int(rng.choice(cur.size, p=inv))
            rem_f = int(cur[rem_idx])
            add_scores = np.array([score_map.get(int(f), 0.0) for f in missing], dtype=float)
            add_scores = np.maximum(add_scores, 1e-12)
            add_scores /= add_scores.sum()
            add_f = int(rng.choice(missing, p=add_scores))
            new_set = set(map(int, cur.tolist()))
            new_set.discard(rem_f)
            new_set.add(add_f)
            g.edge_features[eidx] = np.array(sorted(new_set), dtype=int)

    elif roll < inject_end:
        # Feature injection (constructive).
        tpl = edges_template[eidx]
        tpl_feats = tpl.features.astype(int)
        cur = g.edge_features[eidx].astype(int)
        cur_set = set(map(int, cur.tolist()))
        missing_idx = [i for i, f in enumerate(tpl_feats.tolist()) if int(f) not in cur_set]
        if missing_idx:
            k_add = min(max(1, int(cfg.injection_k)), len(missing_idx))
            if tpl.scores is not None:
                sm = _score_map_for_edge(eidx)
                if sm is None:
                    w = np.array([float(tpl.scores[i]) for i in missing_idx], dtype=float)
                else:
                    w = np.array([float(sm.get(int(tpl_feats[i]), 0.0)) for i in missing_idx], dtype=float)
                w = np.maximum(w, 1e-12)
                w /= w.sum()
                chosen_pos = rng.choice(len(missing_idx), size=k_add, replace=False, p=w)
            else:
                chosen_pos = rng.choice(len(missing_idx), size=k_add, replace=False)
            chosen = [missing_idx[int(p)] for p in np.atleast_1d(chosen_pos)]
            add_feats = tpl_feats[np.array(chosen, dtype=int)]
            merged = np.concatenate([cur, add_feats])
            g.edge_features[eidx] = np.array(sorted(set(map(int, merged.tolist()))), dtype=int)

    # ── 4. Graph contraction ──────────────────────────────────────
    threshold = float(cfg.graph_contraction_threshold) if not isinstance(cfg.graph_contraction_threshold, str) else 0.7
    if feat_similarity is not None and threshold < 1.0:
        feats_c = g.edge_features[eidx]
        if feats_c.size > 1:
            sm_c = _score_map_for_edge(eidx)
            if sm_c is None and scores is not None:
                sm_c = {int(f): float(scores[i])
                        for i, f in enumerate(edges_template[eidx].features.tolist())}
            if sm_c:
                sorted_feats = sorted(feats_c.tolist(),
                                      key=lambda f: sm_c.get(int(f), 0.0), reverse=True)
            else:
                sorted_feats = list(feats_c)
                rng.shuffle(sorted_feats)
            keep = [sorted_feats[0]]
            for f in sorted_feats[1:]:
                if not any(feat_similarity[int(f), int(kf)] >= threshold for kf in keep):
                    keep.append(f)
            g.edge_features[eidx] = np.array(sorted(set(keep)), dtype=int)
            if g.edge_features[eidx].size == 0:
                g.edge_features[eidx] = edges_template[eidx].features[:1].copy()

    # ── 5a. Sparsification (target_feature_ratio) ─────────────────
    if cfg.target_feature_ratio is not None:
        target = float(cfg.target_feature_ratio)
        selected = set()
        for is_on, feats2 in zip(g.active_edges, g.edge_features):
            if bool(is_on):
                selected.update(int(x) for x in feats2.tolist())
        n_ftotal = x_train.shape[1]
        while (len(selected) / max(1, n_ftotal)) > target:
            aidx = np.flatnonzero(g.active_edges)
            if aidx.size == 0:
                break
            idx = int(rng.choice(aidx))
            f2 = g.edge_features[idx]
            if f2.size <= 1:
                if g.active_edges.sum() > 1:
                    g.active_edges[idx] = False
                    selected = set()
                    for is_on, f3 in zip(g.active_edges, g.edge_features):
                        if bool(is_on):
                            selected.update(int(x) for x in f3.tolist())
                else:
                    break
                continue
            sm = _score_map_for_edge(idx)
            if sm:
                drop = min(f2.tolist(), key=lambda f: sm.get(int(f), float("inf")))
            else:
                drop = int(rng.choice(f2))
            g.edge_features[idx] = f2[f2 != drop]
            selected.discard(int(drop))

    # ── 5b. Densification (min_feature_ratio) ─────────────────────
    if cfg.min_feature_ratio is not None:
        min_r = float(cfg.min_feature_ratio)
        n_ftotal = x_train.shape[1]
        selected = set()
        for is_on, feats2 in zip(g.active_edges, g.edge_features):
            if bool(is_on):
                selected.update(int(x) for x in feats2.tolist())
        max_r = float(cfg.target_feature_ratio) if cfg.target_feature_ratio is not None else 1.0
        desired = min(min_r, max_r)
        if (len(selected) / max(1, n_ftotal)) < desired:
            need = int(np.ceil(desired * n_ftotal)) - len(selected)
            aidx = np.flatnonzero(g.active_edges)
            if aidx.size == 0:
                idx = int(rng.integers(0, len(g.active_edges)))
                g.active_edges[idx] = True
                aidx = np.array([idx], dtype=int)
            while need > 0 and aidx.size > 0:
                idx = int(rng.choice(aidx))
                tpl = edges_template[idx]
                tpl_f = tpl.features.astype(int)
                cur = g.edge_features[idx].astype(int)
                cur_set = set(map(int, cur.tolist()))
                miss = [i for i, f in enumerate(tpl_f.tolist()) if int(f) not in cur_set]
                if not miss:
                    inactive = np.flatnonzero(~g.active_edges)
                    if inactive.size > 0:
                        ni = int(rng.choice(inactive))
                        g.active_edges[ni] = True
                        aidx = np.flatnonzero(g.active_edges)
                        g.edge_features[ni] = edges_template[ni].features[:1].copy()
                        selected.update(int(x) for x in g.edge_features[ni].tolist())
                        need = int(np.ceil(desired * n_ftotal)) - len(selected)
                        continue
                    break
                k_add = min(max(1, int(cfg.injection_k)), len(miss), need)
                if tpl.scores is not None:
                    w = np.array([float(tpl.scores[i]) for i in miss], dtype=float)
                    w = np.maximum(w, 1e-12); w /= w.sum()
                    cp = rng.choice(len(miss), size=k_add, replace=False, p=w)
                else:
                    cp = rng.choice(len(miss), size=k_add, replace=False)
                ci = [miss[int(p)] for p in np.atleast_1d(cp)]
                af = tpl_f[np.array(ci, dtype=int)]
                merged = np.array(sorted(set(map(int, np.concatenate([cur, af]).tolist()))), dtype=int)
                g.edge_features[idx] = merged
                selected.update(int(x) for x in af.tolist())
                need = int(np.ceil(desired * n_ftotal)) - len(selected)

    return g


# ═══════════════════════════════════════════════════════════════════
# §13  Repair operator
# ═══════════════════════════════════════════════════════════════════

def repair_hypergraph(
    genome: HypergraphGenome,
    edges_template: list[Hyperedge],
    x_train: sparse.csr_matrix,
    cfg: HypergraphConfig,
    rng: np.random.Generator,
    feat_similarity: np.ndarray | None = None,
) -> HypergraphGenome:
    """Repair: enforce ``target_feature_ratio`` and ``min_feature_ratio``.

    Applied to every offspring (including crossover-only children).
    Graph contraction is NOT applied here—it stays per-edge inside
    the mutation operator.
    """
    g = genome
    n_ftotal = int(x_train.shape[1])

    def _simple_sm(eidx: int) -> dict[int, float] | None:
        s = edges_template[eidx].scores
        if s is None:
            return None
        feats = np.asarray(edges_template[eidx].features, dtype=int)
        return {int(f): max(0.0, float(s[i])) for i, f in enumerate(feats.tolist())} if feats.size else None

    # Sparsification
    if cfg.target_feature_ratio is not None:
        target = float(cfg.target_feature_ratio)
        selected = set()
        for is_on, feats in zip(g.active_edges, g.edge_features):
            if bool(is_on):
                selected.update(int(x) for x in feats.tolist())
        while (len(selected) / max(1, n_ftotal)) > target:
            aidx = np.flatnonzero(g.active_edges)
            if aidx.size == 0:
                break
            idx = int(rng.choice(aidx))
            f2 = g.edge_features[idx]
            if f2.size <= 1:
                if g.active_edges.sum() > 1:
                    g.active_edges[idx] = False
                    selected = set()
                    for is_on, f3 in zip(g.active_edges, g.edge_features):
                        if bool(is_on):
                            selected.update(int(x) for x in f3.tolist())
                else:
                    break
                continue
            sm = _simple_sm(idx)
            if sm:
                drop = min(f2.tolist(), key=lambda f: sm.get(int(f), float("inf")))
            else:
                drop = int(rng.choice(f2))
            g.edge_features[idx] = f2[f2 != drop]
            selected.discard(int(drop))

    # Densification
    if cfg.min_feature_ratio is not None:
        min_r = float(cfg.min_feature_ratio)
        selected = set()
        for is_on, feats in zip(g.active_edges, g.edge_features):
            if bool(is_on):
                selected.update(int(x) for x in feats.tolist())
        max_r = float(cfg.target_feature_ratio) if cfg.target_feature_ratio is not None else 1.0
        desired = min(min_r, max_r)
        if (len(selected) / max(1, n_ftotal)) < desired:
            need = int(np.ceil(desired * n_ftotal)) - len(selected)
            aidx = np.flatnonzero(g.active_edges)
            if aidx.size == 0:
                idx = int(rng.integers(0, len(g.active_edges)))
                g.active_edges[idx] = True
                aidx = np.array([idx], dtype=int)
            while need > 0 and aidx.size > 0:
                idx = int(rng.choice(aidx))
                tpl = edges_template[idx]
                tpl_f = tpl.features.astype(int)
                cur = g.edge_features[idx].astype(int)
                cur_set = set(map(int, cur.tolist()))
                miss = [i for i, f in enumerate(tpl_f.tolist()) if int(f) not in cur_set]
                if not miss:
                    inactive = np.flatnonzero(~g.active_edges)
                    if inactive.size > 0:
                        ni = int(rng.choice(inactive))
                        g.active_edges[ni] = True
                        aidx = np.flatnonzero(g.active_edges)
                        g.edge_features[ni] = edges_template[ni].features[:1].copy()
                        selected.update(int(x) for x in g.edge_features[ni].tolist())
                        need = int(np.ceil(desired * n_ftotal)) - len(selected)
                        continue
                    break
                k_add = min(max(1, int(cfg.injection_k)), len(miss), need)
                if tpl.scores is not None:
                    w = np.array([float(tpl.scores[i]) for i in miss], dtype=float)
                    w = np.maximum(w, 1e-12); w /= w.sum()
                    cp = rng.choice(len(miss), size=k_add, replace=False, p=w)
                else:
                    cp = rng.choice(len(miss), size=k_add, replace=False)
                ci = [miss[int(p)] for p in np.atleast_1d(cp)]
                af = tpl_f[np.array(ci, dtype=int)]
                merged = np.array(sorted(set(map(int, np.concatenate([cur, af]).tolist()))), dtype=int)
                g.edge_features[idx] = merged
                selected.update(int(x) for x in af.tolist())
                need = int(np.ceil(desired * n_ftotal)) - len(selected)
    return g
