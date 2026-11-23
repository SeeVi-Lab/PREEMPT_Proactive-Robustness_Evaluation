"""Trajectory-based robustness analysis of PREEMPT (paper Sec. 3.2-3.3, 5).

All functions take pandas DataFrames and return DataFrames; no file paths.

Input tables
------------
metrics : one row per edited image, columns
    dataset, protection, image_id, chain_id, depth (1..D), concept_similarity
    (+ transform = edit applied at that step, needed only for first-edit grouping).
    ``concept_similarity`` is the paradigm-specific metric S(d): semantic
    similarity for perturbations, bit accuracy for watermarks.
depth0 : one row per protected, unedited image, columns
    dataset, protection, image_id, concept_similarity   (this is S(0)).

Quantities
----------
PSR             S~(d) = S(d) / S(0), per image and chain             (Eq. 1)
survival curve  {S~(0), ..., S~(D)} averaged over images / chains
Protection AUC  (1/D) * sum_d (S~(d) + S~(d+1)) / 2                  (Eq. 3)
clean control   Delta(d) = S_clean(d) - S_prot(d)  on raw S           (Eq. 2)
first-step contribution  alpha = (S~(0) - S~(1)) / (S~(0) - S~(D))    (Sec. 5.5)
rank change     Rank(d=D) - Rank(d=1), rank 1 = highest PSR          (Sec. 5.4)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.paths import normalize_image_id

CLEAN = "none"  # protection name of the unprotected baseline


# ---------------------------------------------------------------- PSR per image
def per_image_psr(metrics: pd.DataFrame, depth0: pd.DataFrame) -> pd.DataFrame:
    """Per-image, per-chain PSR, including a depth-0 row (= 1.0) for every trajectory.

    Returns columns: dataset, protection, image_id, image_stem, chain_id, depth, psr.
    ``image_id`` is taken from ``metrics`` for edited rows and from ``depth0`` for the
    depth-0 rows; ``image_stem`` (file name without extension) links the two.
    Trajectories whose S(0) is missing or zero are dropped.
    """
    m = metrics.dropna(subset=["concept_similarity"]).copy()
    m["depth"] = pd.to_numeric(m["depth"], errors="coerce").astype(int)
    m["image_stem"] = m["image_id"].map(normalize_image_id)

    d0 = depth0.dropna(subset=["concept_similarity"]).copy()
    d0["image_stem"] = d0["image_id"].map(normalize_image_id)
    d0 = d0[d0["concept_similarity"] != 0.0]
    keys = ["dataset", "protection", "image_stem"]
    d0 = d0.drop_duplicates(keys)[keys + ["image_id", "concept_similarity"]].rename(
        columns={"concept_similarity": "s0", "image_id": "image_id_d0"}
    )

    m = m.merge(d0, on=keys, how="inner")
    m["psr"] = m["concept_similarity"] / m["s0"]
    edited = m[keys + ["image_id", "chain_id", "depth", "psr"]]

    start = (
        m[keys + ["image_id_d0", "chain_id"]]
        .drop_duplicates()
        .rename(columns={"image_id_d0": "image_id"})
        .assign(depth=0, psr=1.0)
    )
    return pd.concat([start, edited], ignore_index=True)


def initial_effectiveness(depth0: pd.DataFrame) -> pd.Series:
    """S(0) per protection: mean over images within each dataset, then mean over datasets."""
    d0 = depth0.dropna(subset=["concept_similarity"])
    return d0.groupby(["protection", "dataset"])["concept_similarity"].mean().groupby(level=0).mean()


# ------------------------------------------------------------- survival curves
def survival_curve(psr: pd.DataFrame) -> pd.DataFrame:
    """Survival curve per protection (paper Fig. 3).

    Mean over images within (dataset, chain), then over chains, then over datasets,
    so every dataset and chain has equal weight.
    Returns columns: protection, depth, psr.
    """
    by_chain = psr.groupby(["dataset", "protection", "chain_id", "depth"])["psr"].mean()
    by_dataset = by_chain.groupby(level=["dataset", "protection", "depth"]).mean()
    return by_dataset.groupby(level=["protection", "depth"]).mean().reset_index()


def survival_curve_per_image(psr: pd.DataFrame) -> pd.DataFrame:
    """Survival curve per protection, averaging chains per image first, then over all images.

    Used for ranking (Fig. 5). Returns columns: protection, depth, psr, psr_std.
    """
    per_image = psr.groupby(["dataset", "protection", "image_stem", "depth"])["psr"].mean()
    g = per_image.groupby(level=["protection", "depth"])
    return pd.DataFrame({"psr": g.mean(), "psr_std": g.std().fillna(0.0)}).reset_index()


def paradigm_curve(psr: pd.DataFrame, methods: list[str], name: str) -> pd.DataFrame:
    """Paradigm-level curve (paper Fig. 4).

    For each (dataset, image_id, depth) average PSR over the paradigm's methods and
    chains; then mean and std over these groups. Returns columns: paradigm, depth, psr, psr_std.
    """
    sub = psr[psr["protection"].isin(methods)]
    per_image = sub.groupby(["dataset", "image_id", "depth"])["psr"].mean()
    g = per_image.groupby(level="depth")
    out = pd.DataFrame({"psr": g.mean(), "psr_std": g.std().fillna(0.0)}).reset_index()
    return out.assign(paradigm=name)[["paradigm", "depth", "psr", "psr_std"]]


# ------------------------------------------------------------------- PAUC
def protection_auc(curve: pd.DataFrame, depth_col: str = "depth", value_col: str = "psr") -> float:
    """Eq. 3: trapezoidal area under the survival curve divided by the max depth D."""
    c = curve.sort_values(depth_col)
    d = c[depth_col].to_numpy(dtype=float)
    s = c[value_col].to_numpy(dtype=float)
    if len(d) < 2 or d.max() <= 0:
        return float("nan")
    area = float(np.sum((s[1:] + s[:-1]) / 2.0 * np.diff(d)))
    return area / float(d.max())


def pauc_per_trajectory(psr: pd.DataFrame) -> pd.DataFrame:
    """PAUC of every (dataset, protection, image, chain) trajectory."""
    rows = [
        {"dataset": k[0], "protection": k[1], "image_stem": k[2], "chain_id": k[3], "pauc": protection_auc(c)}
        for k, c in psr.groupby(["dataset", "protection", "image_stem", "chain_id"])
    ]
    return pd.DataFrame(rows)


# -------------------------------------------------------------- clean control
def raw_similarity_curve(metrics: pd.DataFrame) -> pd.DataFrame:
    """Raw S(d) per protection: mean over images per (chain, depth), then mean over chains."""
    m = metrics.dropna(subset=["concept_similarity"]).copy()
    m["depth"] = pd.to_numeric(m["depth"], errors="coerce").astype(int)
    per_chain = m.groupby(["protection", "chain_id", "depth"])["concept_similarity"].mean()
    return per_chain.groupby(level=["protection", "depth"]).mean().rename("s").reset_index()


def clean_control_delta(metrics: pd.DataFrame, clean: str = CLEAN) -> pd.DataFrame:
    """Eq. 2: Delta(d) = S_clean(d) - S_prot(d) for every protected method.

    Only meaningful for perturbation methods (semantic similarity). Returns
    columns: protection, depth, s_clean, s_prot, delta.
    """
    raw = raw_similarity_curve(metrics)
    s_clean = raw[raw["protection"] == clean][["depth", "s"]].rename(columns={"s": "s_clean"})
    prot = raw[raw["protection"] != clean].rename(columns={"s": "s_prot"})
    out = prot.merge(s_clean, on="depth", how="left")
    out["delta"] = out["s_clean"] - out["s_prot"]
    return out[["protection", "depth", "s_clean", "s_prot", "delta"]]


# ----------------------------------------------------- first-step contribution
def first_step_contribution(psr: pd.DataFrame, depth_end: int) -> pd.DataFrame:
    """alpha per protection (paper Tab. 5).

    Per image: average the chains' PSR curves, compute
    alpha = (S~(0) - S~(1)) / (S~(0) - S~(D)); images with S~(0) - S~(D) <= 0
    are skipped. Then mean over images. Returns columns: protection, alpha, alpha_std, n_images.
    """
    per_image = psr.groupby(["protection", "dataset", "image_stem", "depth"])["psr"].mean().unstack("depth")
    per_image = per_image.dropna(subset=[0, 1, depth_end])
    denom = per_image[0] - per_image[depth_end]
    alpha = ((per_image[0] - per_image[1]) / denom)[denom > 0]
    g = alpha.groupby(level="protection")
    return pd.DataFrame(
        {"alpha": g.mean(), "alpha_std": g.std().fillna(0.0), "n_images": g.count()}
    ).reset_index()


# ---------------------------------------------------------------- ranking
def rank_change(curve: pd.DataFrame, methods: list[str], depth_a: int, depth_b: int) -> pd.DataFrame:
    """Rank(d=depth_b) - Rank(d=depth_a) among ``methods`` (rank 1 = highest PSR; ties share the min rank).

    ``curve`` has columns protection, depth, psr. Positive change = the method drops in rank.
    Returns columns: protection, psr_a, rank_a, psr_b, rank_b, rank_change.
    """
    c = curve[curve["protection"].isin(methods)].set_index(["protection", "depth"])["psr"].unstack("depth")
    out = pd.DataFrame(
        {
            "psr_a": c[depth_a],
            "rank_a": c[depth_a].rank(ascending=False, method="min").astype(int),
            "psr_b": c[depth_b],
            "rank_b": c[depth_b].rank(ascending=False, method="min").astype(int),
        }
    )
    out["rank_change"] = out["rank_b"] - out["rank_a"]
    return out.reset_index()


# ------------------------------------------------------ first-edit grouping
def tag_first_edit(psr: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Add ``first_edit`` (the edit applied at depth 1) to every PSR row of the trajectory."""
    m = metrics.copy()
    m["image_stem"] = m["image_id"].map(normalize_image_id)
    first = (
        m[pd.to_numeric(m["depth"], errors="coerce") == 1]
        .groupby(["image_stem", "chain_id", "protection"])["transform"]
        .first()
        .rename("first_edit")
        .reset_index()
    )
    return psr.merge(first, on=["image_stem", "chain_id", "protection"], how="left")


def first_edit_curve(tagged: pd.DataFrame, first_edits: tuple[str, ...], methods: list[str]) -> pd.DataFrame:
    """Mean PSR curve of trajectories whose first edit is in ``first_edits`` (paper Fig. 6).

    Mean over trajectories per method, then mean and std across methods.
    Returns columns: depth, psr, psr_std.
    """
    sub = tagged[tagged["first_edit"].isin(first_edits) & tagged["protection"].isin(methods)]
    by_method = sub.groupby(["protection", "depth"])["psr"].mean()
    g = by_method.groupby(level="depth")
    return pd.DataFrame({"psr": g.mean(), "psr_std": g.std().fillna(0.0)}).reset_index()


# ------------------------------------------ one-shot vs long-horizon
def oneshot_score(oneshot_metrics: pd.DataFrame) -> pd.DataFrame:
    """Per-image one-shot score: mean concept similarity over the semantic edit types."""
    m = oneshot_metrics.dropna(subset=["concept_similarity"]).copy()
    m["image_stem"] = m["image_id"].map(normalize_image_id)
    return (
        m.groupby(["dataset", "protection", "image_stem"])["concept_similarity"]
        .mean()
        .rename("oneshot")
        .reset_index()
    )


def oneshot_vs_pauc(oneshot_metrics: pd.DataFrame, psr: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, dict]:
    """Does one-shot robustness predict long-horizon PAUC? (paper Sec. 5.4)

    Points = (method, chain): mean one-shot score vs mean PAUC over images.
    Returns the points and {"n", "pearson_r", "r2"} of a linear fit.
    """
    joined = pauc_per_trajectory(psr).merge(
        oneshot_score(oneshot_metrics), on=["dataset", "protection", "image_stem"], how="inner"
    )
    joined = joined[joined["protection"].isin(methods)]
    points = joined.groupby(["protection", "chain_id"], as_index=False).agg(
        oneshot=("oneshot", "mean"), pauc=("pauc", "mean"), n_images=("image_stem", "nunique")
    )
    x, y = points["oneshot"].to_numpy(float), points["pauc"].to_numpy(float)
    r = float(np.corrcoef(x, y)[0, 1]) if len(points) >= 3 else float("nan")
    return points, {"n": int(len(points)), "pearson_r": r, "r2": r * r}
