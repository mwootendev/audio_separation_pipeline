#!/usr/bin/env python3
#!/usr/bin/env python3
from __future__ import annotations

from scipy.signal import resample_poly
from math import gcd

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
import logging

import numpy as np

# Optional dependencies; install if missing:
#   pip install soundfile scipy
import soundfile as sf
from scipy.signal import stft, istft, butter, sosfilt
from audio_separator.separator.ensembler import Ensembler


@dataclass
class VocalScore:
    path: Path
    score: float
    details: dict


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def _peak_normalize(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = np.max(np.abs(x)) + 1e-12
    return x * (peak / m)


def _bandpass(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    lo = max(10.0, lo)
    hi = min(0.49 * sr, hi)
    sos = butter(6, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, x)


def _highpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    cutoff = min(0.49 * sr, max(10.0, cutoff))
    sos = butter(6, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, x)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _spectral_flatness(mag: np.ndarray) -> float:
    # mag: (freq, time)
    mag = np.maximum(mag, 1e-12)
    geo = np.exp(np.mean(np.log(mag)))
    ar = np.mean(mag)
    return float(geo / (ar + 1e-12))


def score_vocals(
    vocals: np.ndarray,
    instrumental: Optional[np.ndarray],
    sr: int,
) -> Tuple[float, dict]:
    """
    Reference-free heuristic score. Higher is better.

    Signals we want:
      - strong energy in speech band (~150-8k), esp 300-4k
      - low low-end rumble (<=120 Hz)
      - not too much hiss (very high band)
      - low correlation with instrumental (less leakage)
      - lower spectral flatness in vocal band (less 'noisy'/washy)
    """
    v = _to_mono(vocals).astype(np.float32)
    v = _peak_normalize(v)

    # Bands
    v_speech = _bandpass(v, sr, 150, 8000)
    v_formant = _bandpass(v, sr, 300, 4000)
    v_rumble = _bandpass(v, sr, 20, 120)
    v_hiss = _highpass(v, sr, 9000)

    speech_rms = _rms(v_speech)
    formant_rms = _rms(v_formant)
    rumble_rms = _rms(v_rumble)
    hiss_rms = _rms(v_hiss)

    # STFT for flatness
    f, t, Z = stft(v_formant, fs=sr, nperseg=2048, noverlap=1536, window="hann", padded=False, boundary=None)
    mag = np.abs(Z)
    flat = _spectral_flatness(mag)

    # Leakage penalty vs instrumental (if available)
    corr_pen = 0.0
    if instrumental is not None:
        inst = _to_mono(instrumental).astype(np.float32)
        n = min(len(v), len(inst))
        if n > 0:
            a = v[:n]
            b = inst[:n]
            a = a - np.mean(a)
            b = b - np.mean(b)
            denom = (np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b)) + 1e-12)
            corr = float(np.sum(a * b) / denom)
            corr_pen = abs(corr)  # closer to 0 is better

    # Construct score
    #   reward: speech + formant energy
    #   penalize: rumble proportion, hiss proportion, flatness, correlation leakage
    rumble_ratio = rumble_rms / (speech_rms + 1e-12)
    hiss_ratio = hiss_rms / (speech_rms + 1e-12)

    # Soft log rewards so loud stems don't dominate
    reward = math.log10(1.0 + 20.0 * speech_rms) + 0.8 * math.log10(1.0 + 20.0 * formant_rms)
    penalty = 2.5 * rumble_ratio + 1.8 * hiss_ratio + 1.5 * flat + 2.0 * corr_pen

    score = reward - penalty

    return score, {
        "speech_rms": speech_rms,
        "formant_rms": formant_rms,
        "rumble_ratio": rumble_ratio,
        "hiss_ratio": hiss_ratio,
        "flatness": flat,
        "corr_pen": corr_pen,
        "reward": reward,
        "penalty": penalty,
    }


def rank_vocal_candidates(
    candidates: List[Path],
    instrumentals_by_candidate: dict[Path, Path] | None,
    sr_target: int = 48000,
) -> List[VocalScore]:
    scores: List[VocalScore] = []
    for vpath in candidates:
        v, sr = sf.read(str(vpath), always_2d=True)
        if sr != sr_target:
            # If sample rates differ, keep it simple: score at native rate.
            pass

        ipath = instrumentals_by_candidate.get(vpath) if instrumentals_by_candidate else None
        inst = None
        if ipath and ipath.exists():
            inst, _ = sf.read(str(ipath), always_2d=True)

        sc, det = score_vocals(v, inst, sr)
        scores.append(VocalScore(vpath, sc, det))

    scores.sort(key=lambda s: s.score, reverse=True)
    return scores


def _read_wav_stereo(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=True)
    return x.astype(np.float32), sr


def _read_wav_mono(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = _read_wav_stereo(path)
    x = _to_mono(x).astype(np.float32)
    return x, sr

def _resample_to(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    g = gcd(sr_from, sr_to)
    up = sr_to // g
    down = sr_from // g
    # resample_poly is high-quality and efficient
    axis = 0 if x.ndim > 1 else -1
    return resample_poly(x, up, down, axis=axis).astype(np.float32)


def _match_channels(x: np.ndarray, target_channels: int) -> np.ndarray:
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == target_channels:
        return x
    if x.shape[1] == 1 and target_channels > 1:
        return np.repeat(x, target_channels, axis=1)
    if target_channels == 1 and x.shape[1] > 1:
        return _to_mono(x)[:, None]
    raise ValueError(f"Unsupported channel count {x.shape[1]} for target {target_channels}.")

def build_vocals_ensemble(
    vocal_paths: List[Path],
    scores: dict[Path, float],
    out_path: Path,
    sr: Optional[int] = None,
    top_k: int = 5,
    nperseg: int = 4096,
    noverlap: int = 3072,
    p: float = 1.0,
    combine: str = "avg",  # "avg" (weighted), "max", "min"
    force_mono: bool = False,
) -> Path:
    """
    [DEPRECATED] Weighted complex-STFT merge.
    Delegates to Ensembler utility from python-audio-separator.
    """
    warnings.warn(
        "build_vocals_ensemble is deprecated and will be removed. "
        "Use Ensembler from python-audio-separator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not vocal_paths:
        raise ValueError("No vocal paths provided.")

    # Choose top_k by score
    sorted_paths = sorted(vocal_paths, key=lambda pth: scores.get(pth, -1e9), reverse=True)[:top_k]

    sigs: List[np.ndarray] = []
    srs: List[int] = []
    for vp in sorted_paths:
        x, s = _read_wav_stereo(vp)
        sigs.append(x)
        srs.append(s)

    sr_use = sr or srs[0]
    # Resample if mismatch
    for idx, (x, s) in enumerate(zip(sigs, srs)):
        if s != sr_use:
            sigs[idx] = _resample_to(x, s, sr_use)

    # Normalize channels
    if force_mono:
        sigs = [_to_mono(x)[:, None] if x.ndim > 1 else x[:, None] for x in sigs]
        ch_target = 1
    else:
        ch_target = max(x.shape[1] for x in sigs)
        sigs = [_match_channels(x, ch_target) for x in sigs]

    # Convert to (channels, length) for Ensembler
    waveforms = [x.T for x in sigs]

    # Softmax scores -> per-model scalar weights
    raw = np.array([scores.get(vp, 0.0) for vp in sorted_paths], dtype=np.float64)
    raw = raw - np.max(raw)
    w_model = np.exp(raw)
    w_model = w_model / (np.sum(w_model) + 1e-12)
    w_model = w_model.tolist()

    # Map combine values to Ensembler algorithms
    combine_alg = combine.lower()
    if combine_alg == "avg":
        combine_alg = "avg_fft"
    elif combine_alg == "max":
        combine_alg = "max_fft"
    elif combine_alg == "min":
        combine_alg = "min_fft"

    logger = logging.getLogger("audio_separator")
    ensembler = Ensembler(logger, algorithm=combine_alg, weights=w_model)
    ensembled_wav = ensembler.ensemble(waveforms)

    # Convert back to (length, channels)
    y = ensembled_wav.T
    y = _peak_normalize(y, peak=0.99)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, sr_use)
    return out_path


def build_instrumental_from_mix_minus_vocals(
    mix_path: Path,
    vocals_path: Path,
    out_path: Path,
) -> Path:
    mix, sr1 = _read_wav_stereo(mix_path)
    vox, sr2 = _read_wav_stereo(vocals_path)

    # Resample vocals to mix SR (so mix - vocals is valid)
    if sr1 != sr2:
        vox = _resample_to(vox, sr2, sr1)

    vox = _match_channels(vox, mix.shape[1])
    n = max(mix.shape[0], vox.shape[0])
    mix = np.pad(mix, ((0, n - mix.shape[0]), (0, 0)))
    vox = np.pad(vox, ((0, n - vox.shape[0]), (0, 0)))

    inst = mix - vox
    inst = _peak_normalize(inst, peak=0.99)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), inst, sr1)
    return out_path


def build_instrumental_ensemble(
    instrumental_paths: List[Path],
    out_path: Path,
    mix_path: Optional[Path] = None,
    vocals_path: Optional[Path] = None,
    combine: str = "median",  # "median", "avg", "max", "min"
    force_mono: bool = False,
) -> Path:
    """
    [DEPRECATED] Build an instrumental ensemble that reduces vocal bleed.
    Delegates to Ensembler utility from python-audio-separator.
    """
    warnings.warn(
        "build_instrumental_ensemble is deprecated and will be removed. "
        "Use Ensembler from python-audio-separator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not instrumental_paths:
        raise ValueError("No instrumental paths provided.")

    sigs: List[np.ndarray] = []
    srs: List[int] = []
    sr_target: Optional[int] = None
    if mix_path:
        _, sr_target = sf.read(str(mix_path), always_2d=True)

    for ip in instrumental_paths:
        x, sr = _read_wav_stereo(ip)
        if sr_target is None:
            sr_target = sr
        if sr != sr_target:
            x = _resample_to(x, sr, sr_target)
            sr = sr_target
        sigs.append(x)
        srs.append(sr)

    if sr_target is None:
        sr_target = srs[0]

    if force_mono:
        sigs = [_to_mono(x)[:, None] if x.ndim > 1 else x[:, None] for x in sigs]
        ch_target = 1
    else:
        ch_target = max(x.shape[1] for x in sigs)
        sigs = [_match_channels(x, ch_target) for x in sigs]

    # Convert to (channels, length) for Ensembler
    waveforms = [x.T for x in sigs]

    # Map combine values to Ensembler algorithms (time-domain)
    combine_alg = combine.lower()
    if combine_alg == "avg":
        combine_alg = "avg_wave"
    elif combine_alg == "median":
        combine_alg = "median_wave"
    elif combine_alg == "max":
        combine_alg = "max_wave"
    elif combine_alg == "min":
        combine_alg = "min_wave"

    logger = logging.getLogger("audio_separator")
    ensembler = Ensembler(logger, algorithm=combine_alg)
    ensembled_wav = ensembler.ensemble(waveforms)

    inst = ensembled_wav.T

    if vocals_path:
        voc, sr_v = _read_wav_stereo(vocals_path)
        if sr_v != sr_target:
            voc = _resample_to(voc, sr_v, sr_target)
        voc = _match_channels(voc, inst.shape[1])
        voc = np.pad(voc, ((0, inst.shape[0] - voc.shape[0]), (0, 0)))
        inst = inst - voc

    inst = _peak_normalize(inst, peak=0.99)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), inst, sr_target)
    return out_path
