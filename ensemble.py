#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from stem_ensemble import build_vocals_ensemble, build_instrumental_ensemble


def main():
    ap = argparse.ArgumentParser(description="Build vocal or instrumental ensembles from multiple stems.")
    ap.add_argument("--type", choices=["vocals", "instrumental"], required=True, help="Ensemble mode.")
    ap.add_argument("--combine", choices=["avg", "max", "min", "median"], default="avg", help="Combine strategy (vocals: avg/max/min weighted on mags; instrumental: avg/max/min/median).")
    ap.add_argument("--top-k", type=int, default=None, help="Number of stems to use (vocals only, default: all).")
    ap.add_argument("output", help="Path to output file (e.g., vocals_ensemble.wav).")
    ap.add_argument("inputs", nargs="+", help="Paths to input stem WAV files.")
    args = ap.parse_args()

    output = Path(args.output).resolve()
    inputs = [Path(p).resolve() for p in args.inputs]

    if args.type == "vocals":
        # Equal scores -> equal weights unless more advanced scoring is provided.
        scores = {p: 0.0 for p in inputs}
        top_k = args.top_k or len(inputs)
        build_vocals_ensemble(inputs, scores, output, top_k=min(top_k, len(inputs)), combine=args.combine)
        print(f"Vocals ensemble written to {output}")
    else:
        build_instrumental_ensemble(inputs, output, vocals_path=None, combine=args.combine)
        print(f"Instrumental ensemble written to {output}")


if __name__ == "__main__":
    main()
