#!/usr/bin/env python3
"""
build_dataset.py - stitch features (X) and polarization labels (y) into one table.

The study deliberately splits variables across two files:
  features_corpus.csv      PRE-routing structural features (Fiedler, lambda_max,
                           density, spectral block, angle features) - the X matrix
  polarization_labels.csv  POST-routing label (polarization + sem) AND post-routing
                           quantities (routed_2q, mirror_depth) - the y + mediators

This joins them on the 'file' key so each row is one circuit with BOTH sides
available, ready for the staged regression (stage 1 swaps, stage 2 decay, stage 3
exclusion restriction).

    python build_dataset.py features_corpus.csv polarization_labels.csv full_dataset.csv

Options:
    --keep-unlabeled   left-join instead of inner; keeps circuits with no label
                       (route_fail / unrun) with NaN y. Default is inner (drop them).
"""
import sys, argparse
import pandas as pd
import numpy as np


def build(features_csv, labels_csv, out_csv, keep_unlabeled=False):
    X = pd.read_csv(features_csv)
    y = pd.read_csv(labels_csv)

    n_feat_rows = len(X)
    n_label_rows = len(y)

    # --- restrict to usable rows: 'labeled' (fidelity arm) or 'routed' (swap arm) ---
    if "status" in y.columns:
        ok = {"labeled", "routed"}
        usable = y[y.status.isin(ok)].copy()
        dropped = y[~y.status.isin(ok)]
        if len(dropped):
            from collections import Counter
            reasons = dict(Counter(dropped.get("status", [])))
            print(f"[labels] dropping {len(dropped)} non-usable rows: {reasons}")
    else:
        usable = y.copy()

    # --- identify shared metadata columns to avoid silent collisions ---
    # both tables carry file/algo and an N column; join ONLY on 'file', then
    # reconcile the duplicates explicitly so nothing is overwritten.
    shared = [c for c in X.columns if c in usable.columns and c != "file"]
    # rename the label-side copies of shared metadata so X's versions are canonical
    usable = usable.rename(columns={c: f"{c}__label" for c in shared})

    how = "left" if keep_unlabeled else "inner"
    # one_to_many: X has one row per circuit; labels may have several per circuit
    # (one per device in a multi-device run). Each label row gets its circuit's
    # features. validate guards against accidental feature-side duplication.
    full = X.merge(usable, on="file", how=how, validate="one_to_many")

    # --- sanity: confirm the shared metadata agrees where both exist ---
    for c in shared:
        lc = f"{c}__label"
        if lc in full.columns:
            both = full.dropna(subset=[lc])
            mism = (both[c].astype(str) != both[lc].astype(str)).sum()
            if mism:
                print(f"[WARN] {mism} rows where {c} disagrees between X and y "
                      f"(should be 0 - same circuit). Investigate before trusting join.")
            else:
                full = full.drop(columns=[lc])  # identical -> drop the dup

    # --- report ---
    n_out = len(full)
    n_labeled = full["polarization"].notna().sum() if "polarization" in full else n_out
    print(f"[join] features={n_feat_rows} rows, labels(usable)={len(usable)} rows "
          f"-> joined={n_out} rows ({n_labeled} with a polarization label) [how={how}]")
    unmatched_feat = n_feat_rows - n_out if how == "inner" else (full["polarization"].isna().sum() if "polarization" in full else 0)
    if how == "inner" and n_out < len(usable):
        miss = set(usable["file"]) - set(full["file"])
        print(f"[join] {len(miss)} labeled circuits had NO matching feature row "
              f"(feature extraction range may not cover them): e.g. {list(miss)[:3]}")

    full.to_csv(out_csv, index=False)
    print(f"[done] wrote {n_out} rows x {full.shape[1]} cols -> {out_csv}")
    return full


def main():
    ap = argparse.ArgumentParser(description="Join pre-routing features to polarization labels on 'file'.")
    ap.add_argument("features_csv")
    ap.add_argument("labels_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--keep-unlabeled", action="store_true",
                    help="left-join (keep circuits with no label, y=NaN)")
    args = ap.parse_args()
    build(args.features_csv, args.labels_csv, args.out_csv,
          keep_unlabeled=args.keep_unlabeled)


if __name__ == "__main__":
    main()
