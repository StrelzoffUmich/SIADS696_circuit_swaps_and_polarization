#!/usr/bin/env python3
"""
row_graph.py — reconstruct / visualize the interaction graph of any dataset row.

Reads the `interaction_edges` column (a JSON edge list emitted by extract_features.py)
and rebuilds the exact pre-routing interaction graph for a given circuit, so you can
inspect or draw any row of full_dataset.csv / features.csv.

As a library:
    from row_graph import graph_for
    g = graph_for("full_dataset.csv", "qaoa_indep_none_5_s4.qasm")   # -> networkx.Graph

As a CLI:
    python row_graph.py full_dataset.csv qaoa_indep_none_5_s4.qasm          # print
    python row_graph.py full_dataset.csv qaoa_indep_none_5_s4.qasm --draw out.png
"""
from __future__ import annotations
import json, sys, argparse
import pandas as pd
import networkx as nx


def _row(csv_path, file_id):
    df = pd.read_csv(csv_path)
    if "interaction_edges" not in df.columns:
        sys.exit("CSV has no 'interaction_edges' column — re-run extract_features.py "
                 "with the edge-list addition.")
    hits = df[df["file"] == file_id]
    if hits.empty:
        # allow index or partial match
        if file_id.isdigit():
            hits = df.iloc[[int(file_id)]]
        else:
            hits = df[df["file"].str.contains(file_id, regex=False)]
    if hits.empty:
        sys.exit(f"no row matching '{file_id}' in {csv_path}")
    return hits.iloc[0]


def graph_for(csv_path, file_id):
    """Return the reconstructed networkx interaction graph for a circuit."""
    row = _row(csv_path, file_id)
    edges = json.loads(row["interaction_edges"])
    g = nx.Graph()
    n = int(row.get("n") or row.get("n_qubits") or 0)
    g.add_nodes_from(range(n))           # keep isolated qubits too
    g.add_edges_from(tuple(e) for e in edges)
    g.graph["file"] = row["file"]
    # stash useful stats on the graph for annotation, if present
    for k in ("algo", "n", "fiedler_topology", "routed_2q", "mirror_depth",
              "polarization", "device"):
        if k in row and pd.notna(row[k]):
            g.graph[k] = row[k]
    return g


def draw(g, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    pos = nx.spring_layout(g, seed=42)
    nx.draw_networkx(g, pos, ax=ax, node_color="#4C9AFF", edge_color="#888",
                     node_size=600, font_size=10, font_color="white")
    title = g.graph.get("file", "")
    bits = []
    for k, lbl in [("algo", ""), ("fiedler_topology", "λ₂"),
                   ("routed_2q", "r2q"), ("polarization", "S"), ("device", "")]:
        if k in g.graph:
            v = g.graph[k]
            v = f"{v:.3f}" if isinstance(v, float) else v
            bits.append(f"{lbl}={v}" if lbl else str(v))
    ax.set_title(title + "\n" + "  ".join(bits), fontsize=9)
    ax.axis("off")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("file_id", help="filename (or substring, or row index) to look up")
    ap.add_argument("--draw", metavar="OUT.png", help="render the graph to a PNG")
    args = ap.parse_args()
    g = graph_for(args.csv, args.file_id)
    print(f"{g.graph.get('file')}: {g.number_of_nodes()} qubits, "
          f"{g.number_of_edges()} edges")
    print("edges:", sorted(map(tuple, g.edges())))
    for k in ("algo", "n", "fiedler_topology", "routed_2q", "polarization", "device"):
        if k in g.graph:
            print(f"  {k}: {g.graph[k]}")
    if args.draw:
        draw(g, args.draw)


if __name__ == "__main__":
    main()
