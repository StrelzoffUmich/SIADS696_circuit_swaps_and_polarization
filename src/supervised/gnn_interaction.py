#!/usr/bin/env python3
"""
Interaction-graph GraphSAGE -- the GNN row of the harness comparison.

Reproducible from full_dataset alone: graphs are built from the `interaction_edges`
column (the unique 2-qubit interaction edge set), so there is NO QASM dependency, no
external script to import, and no environment round-trip. This is a self-contained graph
baseline we control, NOT a reproduction of Steven's QASM-trained run -- if his exact
training numbers are wanted, those are produced by his own script and reported separately.

Why interaction_edges loses nothing important: Steven's QASM builder recovers the same 2q
interaction topology, plus per-edge multiplicity (gate-repeat counts). The edge list here is
deduplicated (no per-edge multiplicity), but the multiplicity signal is already summarized in
dataset columns (edge_weight_mean_2q, edge_weight_max_2q, gini_2q_multiplicity, the *_2q_weighted
spectral features), which are fed as graph-level features. So the net sees the same topology and
the same weighted-graph information, sourced cleanly.

Targets: any of route / pol / z -- the harness sets graph.y per fold, so the same module serves
routing AND polarization (we control the head; nothing is locked to a routing-only design).

Architecture: 3 SAGEConv (hidden=128) + LayerNorm + residual + JumpingKnowledge, mean-pooled and
concatenated with gate- and graph-level features, single regression head. Train-fold-only
standardization of features and target. Sized and trained (early-stopped, lr-scheduled) so the
GNN is a fair representative, not an underfit straw man.
"""
from __future__ import annotations
import json, numpy as np, torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, JumpingKnowledge

# Graph-level context fed alongside the pooled node embedding. Includes the spectral features
# RF relies on AND the 2q-multiplicity aggregates that stand in for per-edge weights.
GRAPH_FEATS = ['num_qubits', 'depth', 'size', 'num_2q_gates', 'graph_density',
               'fiedler_topology', 'log_spanning_trees', 'spectral_entropy_topology',
               'fiedler_at_half_depth', 'time_to_connected', 'twoq_temporal_locality',
               'critical_depth', 'parallelism',
               'edge_weight_mean_2q', 'edge_weight_max_2q', 'gini_2q_multiplicity',
               'fiedler_2q_weighted', 'spectral_entropy_2q_weighted']


def build_graphs(df, target):
    graphs = []
    for r in df.itertuples():
        edges = json.loads(r.interaction_edges)
        n = int(r.num_qubits)
        deg = np.zeros(n); el = []
        for u, v in edges:
            el += [[u, v], [v, u]]; deg[u] += 1; deg[v] += 1
        ei = (torch.tensor(el, dtype=torch.long).t().contiguous()
              if el else torch.empty((2, 0), dtype=torch.long))
        x = torch.zeros((n, 3))
        x[:, 0] = torch.tensor(deg)
        x[:, 1] = torch.tensor(deg / max(1, n - 1))
        x[:, 2] = torch.tensor(deg - deg.mean())
        gate = torch.tensor([[getattr(r, 'gate_entropy', 0.0), getattr(r, 'num_unique_gates', 0.0),
                              np.log1p(r.num_2q_gates)]], dtype=torch.float)
        ga = torch.tensor([[float(getattr(r, f, 0.0)) for f in GRAPH_FEATS]], dtype=torch.float)
        y = torch.tensor([[float(getattr(r, target))]], dtype=torch.float)
        graphs.append(Data(x=x, edge_index=ei, y=y, gate_attr=gate, graph_attr=ga))
    return graphs


class Net(torch.nn.Module):
    def __init__(self, n_gate, n_graph, hidden=128, dropout=0.2):
        super().__init__()
        self.c1 = SAGEConv(3, hidden); self.c2 = SAGEConv(hidden, hidden); self.c3 = SAGEConv(hidden, hidden)
        self.n1, self.n2, self.n3 = (torch.nn.LayerNorm(hidden) for _ in range(3))
        self.jk = JumpingKnowledge('cat')
        self.l1 = torch.nn.Linear(hidden * 3 + n_gate + n_graph, 256); self.l2 = torch.nn.Linear(256, 1)
        self.dropout = dropout

    def forward(self, d):
        x, ei, b = d.x, d.edge_index, d.batch
        x1 = F.relu(self.n1(self.c1(x, ei)))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.relu(self.n2(self.c2(x1, ei))) + x1
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x3 = F.relu(self.n3(self.c3(x2, ei))) + x2
        h = global_mean_pool(self.jk([x1, x2, x3]), b)
        h = torch.cat([h, d.gate_attr, d.graph_attr], 1)
        h = F.relu(self.l1(h)); h = F.dropout(h, p=self.dropout, training=self.training)
        return self.l2(h)


class GNNRegressor:
    """fit(graphs, train_idx, val_idx) / predict(graphs, test_idx). Standardizes graph_attr,
    gate_attr, and target on the training fold only -- runs through the harness's group-CV and
    leave-one-unit-out exactly like the sklearn models."""
    def __init__(self, epochs=120, lr=1e-3, patience=15, hidden=128):
        self.epochs, self.lr, self.patience, self.hidden = epochs, lr, patience, hidden

    def _standardize(self, graphs, idx):
        GA = torch.cat([graphs[i].graph_attr for i in idx]); GT = torch.cat([graphs[i].gate_attr for i in idx])
        Y = torch.cat([graphs[i].y for i in idx])
        self.ga_m, self.ga_s = GA.mean(0), GA.std(0).clamp_min(1e-6)
        self.gt_m, self.gt_s = GT.mean(0), GT.std(0).clamp_min(1e-6)
        self.y_m, self.y_s = Y.mean(0), Y.std(0).clamp_min(1e-6)

    def _norm_x(self, d):
        d = d.clone()
        d.graph_attr = (d.graph_attr - self.ga_m) / self.ga_s
        d.gate_attr = (d.gate_attr - self.gt_m) / self.gt_s
        return d

    def _norm(self, d):
        d = self._norm_x(d); d.y = (d.y - self.y_m) / self.y_s
        return d

    def fit(self, graphs, train_idx, val_idx):
        self._standardize(graphs, train_idx)
        tr = [self._norm(graphs[i]) for i in train_idx]; va = [self._norm(graphs[i]) for i in val_idx]
        self.model = Net(graphs[0].gate_attr.shape[1], graphs[0].graph_attr.shape[1], self.hidden)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6)
        crit = torch.nn.SmoothL1Loss()
        tl = DataLoader(tr, batch_size=64, shuffle=True); vl = DataLoader(va, batch_size=128)
        best, bs, bad = 1e9, None, 0
        for ep in range(self.epochs):
            self.model.train()
            for b in tl:
                opt.zero_grad(); loss = crit(self.model(b), b.y.view(-1, 1)); loss.backward(); opt.step()
            self.model.eval(); vloss = 0.0
            with torch.no_grad():
                for b in vl: vloss += crit(self.model(b), b.y.view(-1, 1)).item() * b.num_graphs
            vloss /= len(va); sch.step(vloss)
            if vloss < best - 1e-5: best, bs, bad = vloss, {k: v.clone() for k, v in self.model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience: break
        self.model.load_state_dict(bs); return self

    def predict(self, graphs, test_idx):
        self.model.eval(); te = [self._norm_x(graphs[i]) for i in test_idx]
        out = []
        with torch.no_grad():
            for b in DataLoader(te, batch_size=128): out.append(self.model(b).cpu())
        return torch.cat(out).numpy().ravel() * self.y_s.item() + self.y_m.item()
