#!/usr/bin/env python3
"""NON-CANONICAL, INFORMAL side test -- can a better 17th feature replace `assortativity` in
spectral-17? assortativity is the dead weight (|coef|≈0.01, ablation ΔR²≈-0.0002). Drop it, then
try each candidate from the krystian pool (minus leaky raw counts) as the replacement, ranked by
cross-device OOD route R² (the metric that matters), with in-dist CV R², redundancy (max |Spearman|
vs the kept 16) and OOD support-blowup as context. Fixed α=0.001 (the route operating point) for
speed -- exploratory, not a formal result. Imports the harness; modifies nothing."""
from __future__ import annotations
import sys, pathlib, warnings
HERE = pathlib.Path(__file__).resolve(); SUP = HERE.parent.parent; REPO = SUP.parent.parent
sys.path.insert(0, str(SUP))
import supervised_analysis_run as H               # noqa: E402
import numpy as np, pandas as pd                  # noqa: E402
from scipy.stats import spearmanr                 # noqa: E402
from sklearn.pipeline import Pipeline             # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import Lasso            # noqa: E402
from sklearn.model_selection import GroupKFold    # noqa: E402

warnings.filterwarnings("ignore")
LAB = REPO / "data" / "xdev_out" / "labeled"
OOD = ["FakeSherbrooke", "FakeTorino", "FakeBerlin", "FakeBoston"]
ALPHA = 0.001
LEAKY = {"num_qubits", "depth", "size", "num_2q_gates", "bare_routed_2q", "routed_2q"}


def pipe():
    return Pipeline([("s", StandardScaler()), ("m", Lasso(alpha=ALPHA, max_iter=20000))])


def evaluate(train, oods, cols):
    X = train[cols].fillna(0).values; y = train["route"].values
    gkf = GroupKFold(5); cv = []
    for tr, te in gkf.split(np.arange(len(train)), groups=train.grp.values):
        cv.append(H.metrics_full(y[te], pipe().fit(X[tr], y[tr]).predict(X[te]))["R2"])
    p = pipe().fit(X, y)
    r2, rho = [], []
    for o in oods.values():
        yo = o["route"].values; pr = p.predict(o[cols].fillna(0).values)
        m = H.metrics_full(yo, pr); r2.append(m["R2"]); rho.append(m["Spearman"])
    return float(np.mean(cv)), float(np.mean(r2)), float(np.mean(rho))


def blowup(train, oods, col):
    tlo, thi = float(train[col].min()), float(train[col].max()); span = max(thi - tlo, 1e-9)
    b = 0.0
    for o in oods.values():
        b = max(b, max(float(o[col].max()) - thi, tlo - float(o[col].min()), 0.0) / span)
    return b


def main():
    train = H.load_data(H.DEFAULT_SWAP_CSV, H.DEFAULT_POL_CSV)
    oods = {d: H.load_external(str(LAB / f"val_swap_{d}.csv")) for d in OOD
            if (LAB / f"val_swap_{d}.csv").exists()}
    spectral = [c for c in H.FEATURE_SETS["spectral"] if c in train.columns]
    base16 = [c for c in spectral if c != "assortativity"]

    def present(c):
        return c in train.columns and all(c in o.columns for o in oods.values())

    pool = [c for c in H.FEATURE_SETS["krystian"]
            if present(c) and c not in spectral and c not in LEAKY]
    pool = list(dict.fromkeys(pool))

    cv0, oo0, rh0 = evaluate(train, oods, spectral)              # current spectral-17
    cv16, oo16, rh16 = evaluate(train, oods, base16)            # keep-16 (assortativity removed)
    print(f"baseline spectral-17        : in-dist CV R²={cv0:+.4f}  OOD R²={oo0:+.4f}  OOD ρ={rh0:+.3f}")
    print(f"keep-16 (drop assortativity): in-dist CV R²={cv16:+.4f}  OOD R²={oo16:+.4f}  OOD ρ={rh16:+.3f}")
    print(f"\ntrying {len(pool)} candidate 17th features (base16 + candidate), fixed α={ALPHA}:\n")

    rows = []
    for c in pool:
        cv, oo, rh = evaluate(train, oods, base16 + [c])
        red = max(abs(spearmanr(train[c].values, train[f].values).correlation) for f in base16)
        rows.append(dict(feature=c, cv_r2=cv, ood_r2=oo, ood_rho=rh,
                         d_ood=oo - oo16, max_corr=red, blowup=blowup(train, oods, c)))
    df = pd.DataFrame(rows).sort_values("ood_r2", ascending=False).reset_index(drop=True)
    df.to_csv(REPO / "data" / "results" / "feature_swap_search.csv", index=False)

    print(f"  {'candidate':28s}{'CV R²':>8}{'OOD R²':>9}{'ΔOOD':>8}{'OOD ρ':>8}{'maxcorr':>9}{'blowup×':>9}")
    for _, r in df.iterrows():
        flag = "  <- detonates OOD" if r.blowup > 5 else ("  redundant" if r.max_corr > 0.9 else "")
        print(f"  {r.feature:28s}{r.cv_r2:>+8.4f}{r.ood_r2:>+9.4f}{r.d_ood:>+8.4f}"
              f"{r.ood_rho:>+8.3f}{r.max_corr:>9.2f}{r.blowup:>9.1f}{flag}")
    print(f"\n  (baseline-17 OOD R²={oo0:+.4f};  ΔOOD is vs keep-16={oo16:+.4f}.  blowup = OOD push beyond "
          f"train range, in train-spans.)")
    print("DONE_SWAP")


if __name__ == "__main__":
    main()
