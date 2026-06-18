#!/usr/bin/env python3
"""
swap_features.py — SWAP-arm post-routing quantities, NO simulation.

The SWAP arm asks: does pre-routing structure predict routing overhead? That target
is deterministic — it comes entirely from transpilation (coupling-map routing + SWAP
insertion), with NO noisy simulation. So this pass is cheap and scales to the FULL
N range (unlike the fidelity label, which needs an expensive sim that degenerates and
slows down at high N).

For each circuit it transpiles onto the chosen device and records routing counts for:
  - the BARE circuit U          -> bare_routed_2q, bare_routed_depth, bare_n_active
      "how expensive is THIS circuit to route" — the natural cost target, pairs with
      pre-routing structure features.
  - the MIRROR U.P.U^-1         -> mirror_routed_2q, mirror_routed_depth, mirror_n_active
      matches what the fidelity arm actually executes (so SWAP count and polarization
      can be related on the same routed object). NOTE: mirror routing is only ~2x bare
      routing — the transpiler optimizes the whole mirror jointly (seam cancellation,
      non-symmetric SWAP placement), so it is NOT exactly 2x. Both are recorded so the
      relationship can be measured rather than assumed.

AI-assisted (Claude Opus 4.7/4.8), per SIADS 696 disclosure policy.

    python swap_features.py qasm swap_features.csv --n-lo 3 --n-hi 20 --device FakeBrisbane
"""
from __future__ import annotations
import sys, csv, time, random, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "load_qasm"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helpers"))
from loader_v2 import read_qasm
from qiskit import QuantumCircuit, transpile
import qiskit_ibm_runtime.fake_provider as _fake_provider
from frf import strip_with_map


def resolve_device(name):
    cls = getattr(_fake_provider, name, None)
    if cls is None:
        want = name.lower().replace("fake", "")
        for attr in dir(_fake_provider):
            if attr.startswith("Fake") and attr.lower().replace("fake", "") == want:
                cls = getattr(_fake_provider, attr); name = attr; break
    if cls is None:
        avail = sorted(a for a in dir(_fake_provider)
                       if a.startswith("Fake") and not a.endswith("V1"))
        raise SystemExit(f"unknown device '{name}'. available: {avail}")
    return cls(), name


def _route_counts(circ, dev, opt, seed):
    """Transpile and return (routed_2q, routed_depth, n_active). No simulation."""
    t = transpile(circ, dev, optimization_level=opt, seed_transpiler=seed)
    r2q = sum(1 for op in t.data if len(op.qubits) == 2)
    depth = t.depth()
    ts, _ = strip_with_map(t)
    return r2q, depth, ts.num_qubits


def swap_one(qc, dev, opt, seed, rng):
    base = qc.remove_final_measurements(inplace=False)
    N = base.num_qubits
    rec = dict(n_qubits=N)

    # --- bare circuit routing (the cost target) ---
    bare = base.copy(); bare.measure_all()
    try:
        b2q, bdepth, bN = _route_counts(bare, dev, opt, seed)
    except Exception as e:
        return dict(rec, status="route_fail", err=type(e).__name__)
    rec.update(bare_routed_2q=b2q, bare_routed_depth=bdepth, bare_n_active=bN,
               pre_routing_depth=base.depth(),
               pre_routing_2q=sum(1 for op in base.data if len(op.qubits) == 2))

    # --- mirror routing (matches what the fidelity arm executes) ---
    mask = [rng.randint(0, 1) for _ in range(N)]
    mid = QuantumCircuit(N)
    for q, b in enumerate(mask):
        if b:
            mid.x(q)
    mcp = base.compose(mid).compose(base.inverse()); mcp.measure_all()
    try:
        m2q, mdepth, mN = _route_counts(mcp, dev, opt, seed)
        rec.update(mirror_routed_2q=m2q, mirror_routed_depth=mdepth, mirror_n_active=mN,
                   mirror_over_bare_2q=(m2q / b2q if b2q else None))
    except Exception as e:
        rec.update(mirror_routed_2q=None, status="bare_only", err=type(e).__name__)
        return rec

    rec["status"] = "routed"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qasm_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--n-lo", type=int, default=3)
    ap.add_argument("--n-hi", type=int, default=20)
    ap.add_argument("--families", default="")
    ap.add_argument("--opt", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="FakeBrisbane")
    args = ap.parse_args()

    fams = set(args.families.split(",")) if args.families else None
    dev, device_name = resolve_device(args.device)
    print(f"[device] routing from {device_name} ({dev.num_qubits} qubits) — NO simulation",
          file=sys.stderr)
    rng = random.Random(args.seed)

    files = []
    for f in sorted(Path(args.qasm_dir).glob("*.qasm")):
        pp = f.stem.split("_"); n = int(pp[-2]); algo = "_".join(pp[:-4])
        if (not fams or algo in fams) and args.n_lo <= n <= args.n_hi:
            files.append((algo, n, f))

    done = set()
    outp = Path(args.out_csv)
    if outp.exists():
        with open(outp) as fh:
            for row in csv.DictReader(fh):
                done.add((row["file"], row.get("device", "")))

    fieldnames = ["file", "algo", "n_qubits", "device", "status",
                  "pre_routing_depth", "pre_routing_2q",
                  "bare_routed_2q", "bare_routed_depth", "bare_n_active",
                  "mirror_routed_2q", "mirror_routed_depth", "mirror_n_active",
                  "mirror_over_bare_2q", "err"]
    write_header = not outp.exists()
    fh = open(outp, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
    if write_header:
        w.writeheader()

    t0 = time.time(); n_done = 0
    for algo, n, f in files:
        if (f.name, device_name) in done:
            continue
        rec = dict(file=f.name, algo=algo, device=device_name)
        try:
            rec.update(swap_one(read_qasm(f), dev, args.opt, args.seed, rng))
        except Exception as e:
            rec.update(status="error", err=type(e).__name__)
        w.writerow(rec); fh.flush(); n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done} routed, last={algo}_{n}, {time.time()-t0:.0f}s",
                  file=sys.stderr)
    fh.close()
    print(f"wrote {n_done} rows -> {args.out_csv} ({time.time()-t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
