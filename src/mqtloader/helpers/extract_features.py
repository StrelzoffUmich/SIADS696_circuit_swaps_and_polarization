#!/usr/bin/env python3
"""
extract_features.py — compute features for the QASM corpus and save to CSV.

One-shot. Run after corpus regeneration; downstream analysis scripts then
read <corpus>/features.csv instead of recomputing from QASM every time.

Output columns: file, algo, level, target, n, seed_idx, then all features
from features.extract() (MQT baseline + candidates).

Default corpus = data/qasm with output data/features.csv. Pass --corpus
data/<dir> to point at an alternate corpus root; QASM is read from
<dir>/qasm (or <dir> directly if it already contains *.qasm), and output
is written to <dir>/features.csv.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent          # helpers/  (features.py lives here)
_ROOT = _HERE.parent                              # package root
sys.path.insert(0, str(_HERE))                    # for: features
sys.path.insert(0, str(_ROOT / "load_qasm"))      # for: loader_v2
import json
from features import extract
from features import interaction_graph
from loader_v2 import read_qasm


NAME_RE = re.compile(
    r"^(?P<algo>[a-z][a-z0-9_]*?)"
    r"_(?P<level>alg|indep|nativegates|mapped)"
    r"_(?P<target>[\w+]+)"
    r"_(?P<n>\d+)"
    r"(?:_s(?P<seed>\d+))?"
    r"\.qasm$"
)


def resolve_paths(corpus: Path) -> tuple[Path, Path]:
    """Return (qasm_dir, out_csv) for the given corpus root.

    - data/qasm                 → reads data/qasm, writes data/features.csv
    - data/corpus_v2            → reads data/corpus_v2/qasm, writes
                                  data/corpus_v2/features.csv
    - data/corpus_v2/qasm       → reads that dir, writes parent/features.csv
    """
    if corpus.is_dir() and any(corpus.glob("*.qasm")) and corpus.name == "qasm":
        return corpus, corpus.parent / "features.csv"
    qasm = corpus / "qasm"
    if qasm.is_dir():
        return qasm, corpus / "features.csv"
    # Fallback: assume corpus itself contains qasm files (e.g. data/qasm root)
    if any(corpus.glob("*.qasm")):
        # legacy layout: data/qasm with sibling data/features.csv
        return corpus, corpus.parent / "features.csv"
    return qasm, corpus / "features.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/qasm"),
                    help="Corpus root (e.g. data/qasm or data/corpus_v2)")
    ap.add_argument("--out", type=Path, default=None,
                    help="explicit output CSV path; if omitted, writes features.csv "
                         "next to the corpus (the standalone default). run_pipeline.py "
                         "passes this to write directly into the timestamped run dir.")
    args = ap.parse_args()

    qasm_dir, default_out = resolve_paths(args.corpus)
    out = args.out if args.out is not None else default_out
    paths = sorted(qasm_dir.glob("*.qasm"))
    if not paths:
        print(f"No QASM files in {qasm_dir} — run loader.py first",
              file=sys.stderr)
        sys.exit(2)
    print(f"Computing features for {len(paths)} circuits in {qasm_dir}...",
          flush=True)
    t_start = time.time()
    rows = []
    for i, path in enumerate(paths, 1):
        m = NAME_RE.match(path.name)
        if not m:
            print(f"  skip (unparseable): {path.name}")
            continue
        meta = m.groupdict()
        qc = read_qasm(path)
        rows.append({
            "file": path.name,
            "algo": meta["algo"],
            "level": meta["level"],
            "target": meta["target"],
            "n": int(meta["n"]),
            "seed_idx": int(meta["seed"]) if meta["seed"] else 0,
            "interaction_edges": json.dumps(sorted(map(sorted,
                interaction_graph(qc, weighted=False, source="cliques").edges()))),
            **extract(qc),
        })
        if i % 25 == 0 or i == len(paths):
            print(f"  [{i}/{len(paths)}] {path.name} "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n_meta = 6  # file, algo, level, target, n, seed_idx
    print(f"\nWrote {out}: {len(df)} rows × {df.shape[1]} cols "
          f"({df.shape[1] - n_meta} feature cols) in "
          f"{time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
