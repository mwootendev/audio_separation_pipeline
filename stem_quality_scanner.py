#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf

from stem_ensemble import score_vocals

VOCAL_KEYWORDS = ["vocal"]
INSTRUMENTAL_KEYWORDS = ["instrumental"]


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float32)
    return np.mean(x, axis=1).astype(np.float32)


def _peak_normalize(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = np.max(np.abs(x)) + 1e-12
    return x * (peak / m)


def score_instrumental(audio: np.ndarray, sr: int) -> float:
    """
    Simple heuristic for instrumental quality:
      - balanced spectrum (not overly mid-heavy like vocals)
      - energy across lows/highs
      - lower vocal-band dominance
      - penalize hiss
    """
    x = _peak_normalize(_to_mono(audio))
    n = len(x)
    if n == 0:
        return -1e9
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag = np.abs(np.fft.rfft(x))
    mag = mag + 1e-12

    def band_energy(lo, hi):
        idx = (freqs >= lo) & (freqs < hi)
        if not np.any(idx):
            return 0.0
        return float(np.sqrt(np.mean(np.square(mag[idx]))))

    e_low = band_energy(20, 180)
    e_mid = band_energy(300, 3500)
    e_high = band_energy(8000, 14000)
    e_all = band_energy(20, freqs[-1] if len(freqs) else 20000)

    # Reward lows/highs, penalize excessive mid dominance and hiss
    mid_ratio = e_mid / (e_all + 1e-12)
    hiss_ratio = e_high / (e_all + 1e-12)
    score = (0.5 * np.log1p(e_low) + 0.5 * np.log1p(e_high) + np.log1p(e_all)) - (1.5 * mid_ratio + 0.5 * hiss_ratio)
    return float(score)


def scan_files(paths: List[Path], mode: str) -> List[tuple[Path, float]]:
    results: List[tuple[Path, float]] = []
    for p in paths:
        try:
            audio, sr = sf.read(str(p), always_2d=True)
            if mode == "vocals":
                score, _ = score_vocals(audio, None, sr)
            elif mode == "instrumental":
                score = score_instrumental(audio, sr)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            results.append((p, score))
        except Exception as exc:
            print(f"[error] Failed scoring {p}: {exc}")
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def collect_inputs(target: Path, recursive: bool, mode: str, auto_filter: bool) -> List[Path]:
    if target.is_file():
        return [target]
    pattern = "**/*.wav" if recursive else "*.wav"
    files = sorted(target.glob(pattern))
    if auto_filter:
        kw = VOCAL_KEYWORDS if mode == "vocals" else INSTRUMENTAL_KEYWORDS
        files = [p for p in files if any(k in p.name.lower() for k in kw)]
    return files


def main():
    ap = argparse.ArgumentParser(description="Score stem quality for WAV files (vocals or instrumental).")
    ap.add_argument("path", help="File or directory to score.")
    ap.add_argument("--mode", choices=["vocals", "instrumental"], default="vocals", help="Which heuristic to apply.")
    ap.add_argument("--recursive", action="store_true", help="Recurse directories when path is a directory.")
    ap.add_argument("--auto-filter", action="store_true", help="Filter files by name keywords for the chosen mode.")
    args = ap.parse_args()

    target = Path(args.path).resolve()
    inputs = collect_inputs(target, args.recursive, args.mode, args.auto_filter)
    if not inputs:
        print("No WAV files found to score.")
        return

    results = scan_files(inputs, args.mode)
    width = max(len(p.name) for p, _ in results) if results else 0
    print(f"Scored {len(results)} file(s) in mode='{args.mode}':")
    for p, s in results:
        print(f"{p.name.ljust(width)}  {s: .4f}  ({p})")


if __name__ == "__main__":
    main()
