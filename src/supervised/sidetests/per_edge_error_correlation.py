#!/usr/bin/env python3
"""NON-CANONICAL side test -- standalone runner for the Brisbane-vs-Sherbrooke per-edge gate-error
comparison (the "measuring that Sherbrooke is a different device" appendix figure).

The figure is a REPORT (appendix) artifact, so its producer lives in the canonical report_artifacts.py
(`fig_per_edge_error`) and writes to the report/ figures folder. This wrapper just calls it, so the
appendix figure can be regenerated on its own without running the whole report.

Mechanism (full write-up in report_artifacts.fig_per_edge_error): Brisbane and Sherbrooke are the same
architecture (Eagle r3 / 127q / ecr) with the identical 144-edge coupling map and a matched MEDIAN 2q
gate error, yet for the SAME physical edge the errors are essentially uncorrelated (per-edge Spearman
~0.14) -- same wiring, same median, different geography of which edges are bad. That per-edge scramble
is the device-level origin of Sherbrooke's within-N rank reordering on pol_z.

Run from src/supervised:  python sidetests/per_edge_error_correlation.py
"""
from __future__ import annotations
import sys, pathlib
HERE = pathlib.Path(__file__).resolve(); SUP = HERE.parent.parent
sys.path.insert(0, str(SUP))
import report_artifacts as R   # noqa: E402

if __name__ == "__main__":
    R.fig_per_edge_error()
