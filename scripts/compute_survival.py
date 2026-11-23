#!/usr/bin/env python3
"""Compute PREEMPT trajectory diagnostics from metric CSVs (tables only, no figures).

Inputs (written by `run_preempt.sh metrics`, `survival`, `sequential-complex-metrics`, and score_watermark.py):
  --homo-metrics     per-step metrics of homogeneous chains C1-C4
  --hetero-metrics   per-step metrics of heterogeneous chains CX1-CX4
  --depth0           S(0) of every protected, unedited image
  --watermark-homo / --watermark-hetero / --watermark-depth0   (optional) same for watermarks
  --oneshot-metrics  (optional) one-shot semantic edit metrics, for the one-shot vs PAUC test

Outputs in --out-dir:
  initial_effectiveness.csv  S(0) per method
  survival_curves.csv        PSR curve per method and chain type            
  paradigm_curves.csv        mean +- std PSR per paradigm                   
  pauc.csv                   Protection AUC per method and chain type       (Eq. 3)
  clean_control_delta.csv    Delta(d) = S_clean(d) - S_prot(d)              (Eq. 2)
  first_step_alpha.csv       alpha per method and chain type                
  rank_change.csv            Rank(d=D) - Rank(d=1) per method and chain type 
  first_edit_curves.csv      PSR by first edit type, heterogeneous chains   
  oneshot_vs_pauc.csv + oneshot_vs_pauc.json                                
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils import survival as sv
from utils.paths import RECORDS_DIR

SEMANTIC_SUFFIX = "_edit"  # semantic edits are named content_edit, attribute_edit, ...


def read_metrics(path: Path | None, editor: str, chains: list[str] | None = None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    if "editor" in df.columns:
        df = df[df["editor"] == editor]
    if chains is not None:
        df = df[df["chain_id"].isin(chains)]
    return df


def edited_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with depth >= 1 (score_watermark.py --with-depth0 also writes depth-0 rows)."""
    return df[pd.to_numeric(df["depth"], errors="coerce") >= 1]


def depth0_rows(df: pd.DataFrame) -> pd.DataFrame:
    """S(0) rows: all rows of a depth-0 table, or the depth == 0 rows of a mixed table."""
    if "depth" not in df.columns:
        return df
    depth = pd.to_numeric(df["depth"], errors="coerce")
    return df[depth == 0] if (depth == 0).any() else df


def chain_psr(metrics: pd.DataFrame, depth0: pd.DataFrame, chain_type: str) -> pd.DataFrame:
    return sv.per_image_psr(metrics, depth0).assign(chain_type=chain_type)


def main() -> None:
    metrics_dir = RECORDS_DIR / "metrics"
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--homo-metrics", type=Path, default=metrics_dir / "level3_metrics.csv")
    p.add_argument("--hetero-metrics", type=Path, default=metrics_dir / "sequential_complex_metrics.csv")
    p.add_argument("--depth0", type=Path, default=metrics_dir / "level3_depth0_metrics.csv")
    p.add_argument("--watermark-homo", type=Path)
    p.add_argument("--watermark-hetero", type=Path)
    p.add_argument("--watermark-depth0", type=Path)
    p.add_argument("--oneshot-metrics", type=Path)
    p.add_argument("--editor", default="ip2p")
    p.add_argument("--homo-chains", nargs="+", default=["C1", "C2", "C3", "C4"])
    p.add_argument("--hetero-chains", nargs="+", default=["CX1", "CX2", "CX3", "CX4"])
    p.add_argument("--clean", default=sv.CLEAN, help="protection name of the unprotected baseline")
    p.add_argument("--depth", type=int, default=4, help="maximum chain depth D")
    p.add_argument("--perturbation-methods", nargs="+", help="perturbation methods to analyse (default: all except clean)")
    p.add_argument("--oneshot-methods", nargs="+", help="methods in the one-shot vs PAUC fit (default: --perturbation-methods)")
    p.add_argument("--rank-methods", nargs="+", help="methods ranked against each other (default: all perturbation rows incl. clean)")
    p.add_argument("--out-dir", type=Path, default=RECORDS_DIR / "survival")
    args = p.parse_args()

    homo = read_metrics(args.homo_metrics, args.editor, args.homo_chains)
    hetero = read_metrics(args.hetero_metrics, args.editor, args.hetero_chains)
    depth0 = read_metrics(args.depth0, args.editor)
    pert_psr = pd.concat([chain_psr(homo, depth0, "homo"), chain_psr(hetero, depth0, "hetero")], ignore_index=True)
    pert_methods = args.perturbation_methods or sorted(m for m in pert_psr["protection"].unique() if m != args.clean)

    psr_all, s0_all = [pert_psr], [sv.initial_effectiveness(depth0)]
    paradigms = [("perturbation", pert_psr, pert_methods)]
    if args.watermark_homo and args.watermark_hetero and args.watermark_depth0:
        wm_d0 = depth0_rows(read_metrics(args.watermark_depth0, args.editor))
        wm_psr = pd.concat(
            [chain_psr(edited_rows(read_metrics(args.watermark_homo, args.editor, args.homo_chains)), wm_d0, "homo"),
             chain_psr(edited_rows(read_metrics(args.watermark_hetero, args.editor, args.hetero_chains)), wm_d0, "hetero")],
            ignore_index=True,
        )
        psr_all.append(wm_psr)
        s0_all.append(sv.initial_effectiveness(wm_d0))
        paradigms.append(("watermark", wm_psr, sorted(wm_psr["protection"].unique())))
    psr = pd.concat(psr_all, ignore_index=True)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    pd.concat(s0_all).rename("s0").reset_index().to_csv(out / "initial_effectiveness.csv", index=False)

    curves = pd.concat([sv.survival_curve(g).assign(chain_type=t) for t, g in psr.groupby("chain_type")])
    curves.to_csv(out / "survival_curves.csv", index=False)

    pd.concat([sv.paradigm_curve(g, methods, name) for name, g, methods in paradigms]).to_csv(
        out / "paradigm_curves.csv", index=False
    )

    pauc = [
        {"protection": prot, "chain_type": t, "pauc": sv.protection_auc(c)}
        for (t, prot), c in curves.groupby(["chain_type", "protection"])
    ]
    pd.DataFrame(pauc).to_csv(out / "pauc.csv", index=False)

    delta = pd.concat([
        sv.clean_control_delta(homo, clean=args.clean).assign(chain_type="homo"),
        sv.clean_control_delta(hetero, clean=args.clean).assign(chain_type="hetero"),
    ])
    delta.to_csv(out / "clean_control_delta.csv", index=False)

    pert_only = pert_psr[pert_psr["protection"].isin(pert_methods)]
    pd.concat([
        sv.first_step_contribution(g, depth_end=args.depth).assign(chain_type=t) for t, g in pert_only.groupby("chain_type")
    ]).to_csv(out / "first_step_alpha.csv", index=False)

    rank_methods = args.rank_methods or sorted(pert_psr["protection"].unique())
    pd.concat([
        sv.rank_change(sv.survival_curve_per_image(g), rank_methods, 1, args.depth).assign(chain_type=t)
        for t, g in pert_psr.groupby("chain_type")
    ]).to_csv(out / "rank_change.csv", index=False)

    if "transform" in hetero.columns:
        tagged = sv.tag_first_edit(pert_only[pert_only["chain_type"] == "hetero"], hetero)
        edits = tagged["first_edit"].dropna().unique()
        semantic = tuple(e for e in edits if e.endswith(SEMANTIC_SUFFIX))
        physical = tuple(e for e in edits if not e.endswith(SEMANTIC_SUFFIX))
        pd.concat([
            sv.first_edit_curve(tagged, physical, pert_methods).assign(first_edit="physical"),
            sv.first_edit_curve(tagged, semantic, pert_methods).assign(first_edit="semantic"),
        ]).to_csv(out / "first_edit_curves.csv", index=False)

    if args.oneshot_metrics:
        points, stats = sv.oneshot_vs_pauc(
            read_metrics(args.oneshot_metrics, args.editor), pert_psr, args.oneshot_methods or pert_methods
        )
        points.to_csv(out / "oneshot_vs_pauc.csv", index=False)
        (out / "oneshot_vs_pauc.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"Wrote tables to {out}")


if __name__ == "__main__":
    main()
