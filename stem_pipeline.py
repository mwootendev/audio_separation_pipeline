#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List
import shutil
import time

import yaml
import requests

from audio_separator.separator import Separator
from stem_ensemble import build_vocals_ensemble, build_instrumental_ensemble


VOCAL_KEYWORDS = ["vocal", "vox", "sing", "voice"]
INSTRUMENTAL_KEYWORDS = ["instrumental", "drums", "bass", "guitar", "piano"]


def load_config(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping.")
    return cfg


def resolve_path(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def collect_inputs(stage_cfg: dict, mix_path: Path, base_output: Path) -> List[Path]:
    inp = stage_cfg.get("input_dir")
    if not inp:
        return []
    if str(inp).upper() == "MIX":
        return [mix_path]
    inp_dir = resolve_path(base_output, inp)
    candidates = sorted(inp_dir.rglob("*.wav")) if inp_dir and inp_dir.exists() else []
    filters = stage_cfg.get("filename_filter")
    if filters:
        if isinstance(filters, str):
            filters = [filters]
        lfilt = [f.lower() for f in filters if f]
        candidates = [p for p in candidates if any(f in p.name.lower() for f in lfilt)]
    return candidates


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p




def _resample_audio(audio: "np.ndarray", sr: int, target_sr: int) -> "np.ndarray":
    from math import gcd
    from scipy.signal import resample_poly

    if sr == target_sr:
        return audio
    g = gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    return resample_poly(audio, up, down, axis=0).astype("float32")

def _resample_inputs(inputs: List[Path], target_sr: int, work_dir: Path) -> List[Path]:
    import soundfile as sf

    work_dir.mkdir(parents=True, exist_ok=True)
    resampled: List[Path] = []
    for idx, wav in enumerate(inputs):
        audio, sr = sf.read(str(wav), always_2d=True)
        if sr == target_sr:
            resampled.append(wav)
            continue
        y = _resample_audio(audio, sr, target_sr)
        out_path = work_dir / f"{wav.stem}_sr{target_sr}_{idx}{wav.suffix}"
        sf.write(str(out_path), y, target_sr)
        resampled.append(out_path)
    return resampled

def _clean_stage_inputs(stage_cfg: dict, base_output: Path, out_dir: Path, inputs: List[Path]) -> None:
    if not stage_cfg.get("clean"):
        return
    inp = stage_cfg.get("input_dir")
    if not inp or str(inp).upper() == "MIX":
        return
    inp_dir = resolve_path(base_output, inp)
    if not inp_dir or not inp_dir.exists():
        return
    base_dir = inp_dir.resolve()
    for p in inputs:
        if not p.is_file():
            continue
        try:
            p.resolve().relative_to(base_dir)
        except Exception:
            continue
        p.unlink(missing_ok=True)
    if stage_cfg.get("filename_filter"):
        return
    if base_dir == out_dir.resolve():
        return
    shutil.rmtree(inp_dir, ignore_errors=True)


def _clean_dict(d: dict | None) -> dict | None:
    """Remove keys with None values, recursively for dicts/lists."""
    if d is None:
        return None
    if not isinstance(d, dict):
        return d
    cleaned = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            v = _clean_dict(v)
        elif isinstance(v, list):
            v = [_clean_dict(x) if isinstance(x, dict) else x for x in v]
        cleaned[k] = v
    return cleaned


def _resolve_mask_ref(stage_cfg: dict, base_output: Path, mix_path: Path) -> Path | None:
    mask_ref = stage_cfg.get("mask_ref")
    if not mask_ref:
        return None
    if str(mask_ref).upper() == "MIX":
        return mix_path
    return resolve_path(base_output, str(mask_ref))


def _build_weight_curve(freqs: "np.ndarray", weights: list | None) -> "np.ndarray":
    import numpy as np

    curve = np.ones_like(freqs, dtype=np.float32)
    if not weights:
        return curve
    for band in weights:
        if not isinstance(band, (list, tuple)) or len(band) < 3:
            continue
        lo, hi, weight = band[:3]
        try:
            lo_hz = float(lo)
            hi_hz = float(hi)
            w = float(weight)
        except Exception:
            continue
        if hi_hz <= lo_hz:
            continue
        idx = (freqs >= lo_hz) & (freqs <= hi_hz)
        curve[idx] *= w
    return np.clip(curve, 0.0, 1.0)


def config_separator(model_cfg: dict, global_cfg: dict, stage_cfg: dict, model_dir: Path, out_dir: Path) -> Separator:
    sep_kwargs: Dict[str, Any] = {}
    # Global defaults
    for key in ["model_file_dir", "output_format", "sample_rate", "use_autocast"]:
        if key in global_cfg and global_cfg[key] is not None:
            sep_kwargs[key] = global_cfg[key]
    # Stage overrides
    for key in ["output_single_stem", "invert_using_spec", "mdx_params", "vr_params", "demucs_params", "mdxc_params"]:
        if key in stage_cfg and stage_cfg[key] is not None:
            sep_kwargs[key] = stage_cfg[key]
    # Model overrides
    for key in ["output_single_stem", "invert_using_spec", "mdx_params", "vr_params", "demucs_params", "mdxc_params"]:
        if key in model_cfg and model_cfg[key] is not None:
            sep_kwargs[key] = model_cfg[key]

    # Clean nested param dicts to strip None values
    for nested in ["mdx_params", "vr_params", "demucs_params", "mdxc_params"]:
        if nested in sep_kwargs:
            sep_kwargs[nested] = _clean_dict(sep_kwargs[nested])

    sep_kwargs["model_file_dir"] = str(model_dir)
    sep_kwargs["output_dir"] = str(out_dir)

    sep = Separator(**sep_kwargs)
    sep.load_model(model_filename=model_cfg["model_file"])
    return sep


def _strip_filename_filters(input_stem: str, filename_filters: list[str] | None) -> str:
    if not filename_filters:
        return input_stem
    lowered = input_stem.lower()
    for filt in filename_filters:
        if not filt:
            continue
        needle = filt.lower()
        while True:
            idx = lowered.find(needle)
            if idx < 0:
                break
            input_stem = input_stem[:idx] + input_stem[idx + len(filt):]
            lowered = input_stem.lower()
    return " ".join(input_stem.split())


def rename_outputs(
    generated: List[Path],
    target_dir: Path,
    input_stem: str,
    model_name: str,
    output_names: dict | None,
    filename_filters: list[str] | None = None,
) -> List[Path]:
    renamed: List[Path] = []
    if output_names and filename_filters:
        input_stem = _strip_filename_filters(input_stem, filename_filters)
    for src in generated:
        src_path = Path(src)
        if not src_path.is_absolute():
            src_path = target_dir / src_path

        stem_type = src_path.stem
        if output_names:
            for k, v in output_names.items():
                if k and k.lower() in src_path.stem.lower():
                    stem_type = v
                    break
        new_name = f"{input_stem}-{model_name} ({stem_type}){src_path.suffix}"
        dest = target_dir / new_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        src_path.rename(dest)
        renamed.append(dest)
    return renamed


def run_stage(stage_cfg: dict, global_cfg: dict, mix_path: Path, base_output: Path, model_dir: Path) -> List[Path]:
    stage_type = stage_cfg.get("type", "separate").lower()
    inputs = collect_inputs(stage_cfg, mix_path, base_output)
    if not inputs:
        print(f"[{stage_cfg.get('name','?')}] No inputs, skipping.")
        return []

    out_dir = ensure_dir(resolve_path(base_output, stage_cfg.get("output_dir") or "."))

    inputs_for_stage = inputs
    resample_dir: Path | None = None
    target_sr = None
    if stage_type in ["ensemble", "filter", "spectral_mask"]:
        target_sr = global_cfg.get("sample_rate")
        if target_sr:
            resample_dir = out_dir / "_work" / "resample"
            inputs_for_stage = _resample_inputs(inputs, int(target_sr), resample_dir)

    outputs: List[Path] = []
    if stage_type == "ensemble":
        outputs = run_ensemble_stage(stage_cfg, out_dir, inputs_for_stage)
    elif stage_type == "filter":
        outputs = run_filter_stage(stage_cfg, out_dir, inputs_for_stage, global_cfg)
    elif stage_type == "spectral_mask":
        outputs = run_spectral_mask_stage(stage_cfg, out_dir, inputs_for_stage, base_output, mix_path, target_sr)
    elif stage_type == "midi":
        outputs = run_midi_stage(stage_cfg, out_dir, inputs)
    elif stage_type == "mvsep":
        outputs = run_mvsep_stage(stage_cfg, global_cfg, out_dir, inputs)
    else:
        # separate
        all_outputs: List[Path] = []
        work_root = out_dir / "_work"

        for model_cfg in stage_cfg.get("models", []):
            friendly = model_cfg.get("name") or Path(model_cfg["model_file"]).stem
            output_names = model_cfg.get("output_names") or stage_cfg.get("output_names")
            model_out_dir = ensure_dir(work_root / friendly)
            sep = config_separator(model_cfg, global_cfg, stage_cfg, model_dir, model_out_dir)
            for audio_file in inputs:
                try:
                    generated = sep.separate(str(audio_file))
                except Exception as exc:
                    print(f"[error] Separation failed for model '{friendly}' on '{audio_file.name}': {exc}")
                    generated = []
                generated_paths = [Path(p) if Path(p).is_absolute() else model_out_dir / p for p in generated]
                if not generated_paths:
                    print(f"[warn] No outputs generated for model '{friendly}' on '{audio_file.name}'.")
                filename_filters = stage_cfg.get("filename_filter")
                if isinstance(filename_filters, str):
                    filename_filters = [filename_filters]
                renamed = rename_outputs(
                    generated_paths,
                    out_dir,
                    audio_file.stem,
                    friendly,
                    output_names,
                    filename_filters=filename_filters,
                )
                all_outputs.extend(renamed)

        if work_root.exists():
            leftovers = [p for p in work_root.rglob("*") if p.is_file()]
            if leftovers:
                print(f"[error] Work folder not empty, removing leftover files: {[str(p) for p in leftovers]}")
            shutil.rmtree(work_root, ignore_errors=True)
        outputs = all_outputs

    if resample_dir and resample_dir.exists():
        shutil.rmtree(resample_dir, ignore_errors=True)

    _clean_stage_inputs(stage_cfg, base_output, out_dir, inputs)
    return outputs


def _get_mvsep_key(stage_cfg: dict, global_cfg: dict) -> str | None:
    return stage_cfg.get("mvsep_api_key") or global_cfg.get("mvsep_api_key")


def _mvsep_create_separation(api_key: str, audio_path: Path, sep_type: int, add_opt1: int | None, add_opt2: int | None) -> str | None:
    print(
        f"[mvsep] create request: file='{audio_path.name}', sep_type={sep_type}, "
        f"add_opt1={add_opt1}, add_opt2={add_opt2}"
    )
    files = {
        "audiofile": audio_path.open("rb"),
        "api_token": (None, api_key),
        "sep_type": (None, str(sep_type)),
        "add_opt1": (None, "" if add_opt1 is None else str(add_opt1)),
        "add_opt2": (None, "" if add_opt2 is None else str(add_opt2)),
        "output_format": (None, "1"),
        "is_demo": (None, "0"),
    }
    try:
        response = requests.post("https://mvsep.com/api/separation/create", files=files, timeout=60)
    finally:
        files["audiofile"].close()

    if response.status_code != 200:
        print(f"[error] MVSEP create failed for {audio_path.name}: {response.status_code}")
        return None

    try:
        parsed = response.json()
        return parsed["data"]["hash"]
    except Exception as exc:
        print(f"[error] MVSEP create parse failed for {audio_path.name}: {exc}")
        return None


def _mvsep_wait_for_files(hash_id: str, poll_interval: float, timeout_seconds: float) -> list[dict] | None:
    start = time.time()
    while True:
        print(f"[mvsep] status request: hash={hash_id}")
        response = requests.get("https://mvsep.com/api/separation/get", params={"hash": hash_id}, timeout=60)
        if response.status_code != 200:
            print(f"[error] MVSEP status failed for hash {hash_id}: {response.status_code}")
            return None
        try:
            data = response.json()
        except Exception as exc:
            print(f"[error] MVSEP status parse failed for hash {hash_id}: {exc}")
            return None

        if not data.get("success"):
            print(f"[error] MVSEP reported failure for hash {hash_id}.")
            return None

        files = (data.get("data") or {}).get("files")
        if files:
            return files

        if time.time() - start > timeout_seconds:
            print(f"[error] MVSEP timeout waiting for hash {hash_id}.")
            return None
        time.sleep(poll_interval)


def _mvsep_filter_files(files: list[dict], output_single_stem: str | None, output_names: dict | None) -> list[dict]:
    if not output_single_stem:
        return files
    desired = str(output_single_stem).strip().lower()
    if not desired:
        return files
    filtered: list[dict] = []
    for file_info in files:
        filename = str(file_info.get("download") or file_info.get("name") or "").lower()
        if not filename:
            continue
        if desired in filename:
            filtered.append(file_info)
            continue
        if output_names:
            for key, value in output_names.items():
                if not key or not value:
                    continue
                if key.lower() in filename and str(value).lower() == desired:
                    filtered.append(file_info)
                    break
    return filtered


def _mvsep_download_files(files: list[dict], dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for file_info in files:
        url = str(file_info.get("url", "")).replace("\\/", "/")
        filename = file_info.get("download") or "mvsep_output.wav"
        if not url:
            continue
        print(f"[mvsep] download request: file='{filename}'")
        response = requests.get(url, timeout=120)
        if response.status_code != 200:
            print(f"[error] MVSEP download failed for {filename}: {response.status_code}")
            continue
        out_path = dest_dir / filename
        out_path.write_bytes(response.content)
        downloaded.append(out_path)
    return downloaded


def run_mvsep_stage(stage_cfg: dict, global_cfg: dict, out_dir: Path, inputs: List[Path]) -> List[Path]:
    api_key = _get_mvsep_key(stage_cfg, global_cfg)
    if not api_key:
        print(f"[{stage_cfg.get('name','?')}] No mvsep_api_key provided, skipping.")
        return []

    poll_interval = float(stage_cfg.get("poll_interval_seconds") or 5.0)
    timeout_seconds = float(stage_cfg.get("timeout_seconds") or 1800.0)

    all_outputs: List[Path] = []
    work_root = out_dir / "_work"

    for model_cfg in stage_cfg.get("models", []):
        friendly = model_cfg.get("name") or f"mvsep_{model_cfg.get('sep_type')}"
        output_names = model_cfg.get("output_names") or stage_cfg.get("output_names")
        output_single_stem = stage_cfg.get("output_single_stem")
        if model_cfg.get("output_single_stem") is not None:
            output_single_stem = model_cfg.get("output_single_stem")
        sep_type = model_cfg.get("sep_type")
        if sep_type is None:
            print(f"[error] MVSEP model '{friendly}' missing sep_type, skipping.")
            continue
        add_opt1 = model_cfg.get("add_opt1")
        add_opt2 = model_cfg.get("add_opt2")
        model_out_dir = ensure_dir(work_root / friendly)

        for audio_file in inputs:
            hash_id = _mvsep_create_separation(api_key, audio_file, int(sep_type), add_opt1, add_opt2)
            if not hash_id:
                continue
            files = _mvsep_wait_for_files(hash_id, poll_interval, timeout_seconds)
            if not files:
                continue
            files = _mvsep_filter_files(files, output_single_stem, output_names)
            if not files:
                print(f"[warn] No MVSEP outputs matched output_single_stem='{output_single_stem}' for model '{friendly}' on '{audio_file.name}'.")
                continue
            downloaded = _mvsep_download_files(files, model_out_dir)
            if not downloaded:
                print(f"[warn] No MVSEP outputs for model '{friendly}' on '{audio_file.name}'.")
                continue
            filename_filters = stage_cfg.get("filename_filter")
            if isinstance(filename_filters, str):
                filename_filters = [filename_filters]
            renamed = rename_outputs(
                downloaded,
                out_dir,
                audio_file.stem,
                friendly,
                output_names,
                filename_filters=filename_filters,
            )
            all_outputs.extend(renamed)

    if work_root.exists():
        leftovers = [p for p in work_root.rglob("*") if p.is_file()]
        if leftovers:
            print(f"[error] MVSEP work folder not empty, removing leftover files: {[str(p) for p in leftovers]}")
        shutil.rmtree(work_root, ignore_errors=True)
    return all_outputs


def run_ensemble_stage(stage_cfg: dict, out_dir: Path, inputs: List[Path]) -> List[Path]:
    mode = stage_cfg.get("mode", "vocals").lower()
    top_k = stage_cfg.get("top_k")
    combine_raw = stage_cfg.get("combine_type") or stage_cfg.get("combine")
    output_name = stage_cfg.get("output_name") or f"{mode}_ensemble.wav"
    if isinstance(combine_raw, (list, tuple)):
        combine_values = [str(v).strip() for v in combine_raw if v is not None and str(v).strip()]
    elif combine_raw is None:
        combine_values = []
    else:
        combine_values = [str(combine_raw).strip()] if str(combine_raw).strip() else []
    if not combine_values:
        combine_values = [None]
    multi_combine = len(combine_values) > 1
    outputs: List[Path] = []
    if mode == "vocals":
        candidates = [p for p in inputs if any(k in p.name.lower() for k in VOCAL_KEYWORDS)]
        scores = {p: 0.0 for p in candidates}
        for combine in combine_values:
            if multi_combine and combine is not None:
                prefix = f"{combine.capitalize()} "
                out_path = out_dir / f"{prefix}{output_name}"
            else:
                out_path = out_dir / output_name
            build_vocals_ensemble(
                candidates,
                scores,
                out_path,
                top_k=min(top_k or len(candidates), len(candidates)),
                combine=(combine or "avg"),
            )
            print(f"[{stage_cfg.get('name','?')}] Vocals ensemble -> {out_path}")
            outputs.append(out_path)
        return outputs
    if mode == "instrumental":
        candidates = [p for p in inputs if any(k in p.name.lower() for k in INSTRUMENTAL_KEYWORDS)]
        for combine in combine_values:
            if multi_combine and combine is not None:
                prefix = f"{combine.capitalize()} "
                out_path = out_dir / f"{prefix}{output_name}"
            else:
                out_path = out_dir / output_name
            combine_use = (combine or "median").lower()
            build_instrumental_ensemble(candidates, out_path, vocals_path=None, combine=combine_use)
            print(f"[{stage_cfg.get('name','?')}] Instrumental ensemble -> {out_path}")
            outputs.append(out_path)
        return outputs
    print(f"[{stage_cfg.get('name','?')}] Unknown ensemble mode '{mode}', skipping.")
    return []


def _normalize_plugins_cfg(plugins_cfg: list | None) -> list[dict]:
    if not plugins_cfg:
        return []
    normalized: list[dict] = []
    for plugin in plugins_cfg:
        if isinstance(plugin, str):
            normalized.append({"type": plugin})
        elif isinstance(plugin, dict):
            normalized.append(dict(plugin))
    return normalized


def _resolve_vst3_path(path_value: str | None, global_cfg: dict) -> Path | None:
    if not path_value:
        return None
    p = Path(path_value)
    if p.is_absolute():
        return p
    base = global_cfg.get("vst3_plugin_dir")
    if base:
        return Path(base) / p
    return p


def _build_plugin(plugin_cfg: dict, global_cfg: dict, sample_rate: int) -> "AudioProcessor":
    from pedalboard import (
        Clipping,
        Compressor,
        Gain,
        HighShelfFilter,
        HighpassFilter,
        Invert,
        LadderFilter,
        LowShelfFilter,
        LowpassFilter,
        NoiseGate,
        PitchShift,
        Resample,
        VST3Plugin,
    )

    plugin_type = str(plugin_cfg.get("type") or plugin_cfg.get("name") or "").strip()
    if not plugin_type:
        raise ValueError("Filter plugin is missing a 'type'.")
    params = {k: v for k, v in plugin_cfg.items() if k not in {"type", "name", "path", "plugin_path", "file", "parameters"}}

    if plugin_type == "VST3Plugin":
        vst_path = (
            plugin_cfg.get("path")
            or plugin_cfg.get("plugin_path")
            or plugin_cfg.get("file")
        )
        resolved = _resolve_vst3_path(vst_path, global_cfg)
        if not resolved:
            raise ValueError("VST3Plugin requires a 'path' or 'plugin_path'.")
        plugin = VST3Plugin(str(resolved), sample_rate=sample_rate)
        parameters = plugin_cfg.get("parameters") or {}
        for name, value in parameters.items():
            try:
                plugin.parameters[name].value = value
            except Exception:
                print(f"[warn] VST3Plugin parameter '{name}' not applied.")
        return plugin

    plugin_map = {
        "Clipping": Clipping,
        "Compressor": Compressor,
        "Gain": Gain,
        "HighShelfFilter": HighShelfFilter,
        "HighpassFilter": HighpassFilter,
        "Invert": Invert,
        "LadderFilter": LadderFilter,
        "LowShelfFilter": LowShelfFilter,
        "LowpassFilter": LowpassFilter,
        "NoiseGate": NoiseGate,
        "PitchShift": PitchShift,
        "Resample": Resample,
    }
    cls = plugin_map.get(plugin_type)
    if not cls:
        raise ValueError(f"Unsupported filter plugin '{plugin_type}'.")
    return cls(**params)


def _apply_plugins(audio: "np.ndarray", sample_rate: int, plugins_cfg: list[dict], global_cfg: dict) -> tuple["np.ndarray", int]:
    from pedalboard import Resample

    current_sr = sample_rate
    for plugin_cfg in plugins_cfg:
        plugin = _build_plugin(plugin_cfg, global_cfg, current_sr)
        audio = plugin(audio, current_sr)
        if isinstance(plugin, Resample):
            target_sr = getattr(plugin, "target_sample_rate", None) or getattr(plugin, "sample_rate", None)
            if target_sr:
                current_sr = int(target_sr)
    return audio, current_sr


def _apply_bandpass_sums(audio: "np.ndarray", sample_rate: int, bands: list) -> "np.ndarray":
    import numpy as np
    from pedalboard import Pedalboard, HighpassFilter, LowpassFilter

    y = np.zeros_like(audio)
    for lo, hi in bands:
        lo_hz = max(10.0, float(lo))
        hi_hz = min(0.49 * sample_rate, float(hi))
        board = Pedalboard([HighpassFilter(cutoff_frequency_hz=lo_hz), LowpassFilter(cutoff_frequency_hz=hi_hz)])
        y += board(audio, sample_rate)
    return y


def run_filter_stage(stage_cfg: dict, out_dir: Path, inputs: List[Path], global_cfg: dict) -> List[Path]:
    """
    Apply pedalboard filter chains to inputs.
    stage_cfg supports:
      bands: list of [lo, hi] Hz ranges to keep (summed via Highpass/Lowpass filters)
      plugins: list of pedalboard plugins to apply in order
    """
    import numpy as np
    from pedalboard.io import AudioFile

    bands = stage_cfg.get("bands") or []
    plugins_cfg = _normalize_plugins_cfg(stage_cfg.get("plugins"))
    if not bands and not plugins_cfg:
        print(f"[{stage_cfg.get('name','?')}] No bands or plugins specified, skipping.")
        return []

    outputs: List[Path] = []
    for wav in inputs:
        with AudioFile(str(wav)) as f:
            audio = f.read(f.frames)
            sr = f.samplerate
        if audio.ndim == 1:
            audio = audio[None, :]
        if audio.shape[0] > audio.shape[1]:
            audio = audio.T

        y = audio.astype(np.float32)
        if bands:
            y = _apply_bandpass_sums(y, sr, bands)
        if plugins_cfg:
            y, sr = _apply_plugins(y, sr, plugins_cfg, global_cfg)

        if stage_cfg.get("normalize", True):
            peak = np.max(np.abs(y)) + 1e-12
            y = y * (0.99 / peak)
        out_path = out_dir / f"{wav.stem} ({stage_cfg.get('name','filter')}).wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with AudioFile(str(out_path), "w", samplerate=sr, num_channels=y.shape[0]) as f:
            f.write(y)
        outputs.append(out_path)
    return outputs


def run_spectral_mask_stage(
    stage_cfg: dict,
    out_dir: Path,
    inputs: List[Path],
    base_output: Path,
    mix_path: Path,
    target_sr: int | None,
) -> List[Path]:
    import numpy as np
    import soundfile as sf
    from scipy.signal import istft, stft, wiener

    mask_ref = _resolve_mask_ref(stage_cfg, base_output, mix_path)
    if not mask_ref or not mask_ref.exists():
        print(f"[{stage_cfg.get('name','?')}] mask_ref not found, skipping.")
        return []

    fft_size = int(stage_cfg.get("fft_size") or 2048)
    win_length = int(stage_cfg.get("win_length") or fft_size)
    hop_length = int(stage_cfg.get("hop_length") or (win_length // 4))
    weights = stage_cfg.get("weights") or []

    soft_cfg = stage_cfg.get("soft_mask") or {}
    soft_enabled = bool(soft_cfg.get("enabled"))
    soft_strength = float(soft_cfg.get("strength") or 0.8)

    wiener_cfg = stage_cfg.get("wiener") or {}
    wiener_enabled = bool(wiener_cfg.get("enabled"))
    wiener_iterations = int(wiener_cfg.get("iterations") or 1)
    wiener_strength = float(wiener_cfg.get("strength") or 0.5)
    wiener_mysize = wiener_cfg.get("mysize") or (5, 5)

    outputs: List[Path] = []
    for wav in inputs:
        audio, sr = sf.read(str(wav), always_2d=True)
        ref_audio, ref_sr = sf.read(str(mask_ref), always_2d=True)
        if target_sr and sr != target_sr:
            audio = _resample_audio(audio, sr, target_sr)
            sr = target_sr
        if ref_sr != sr:
            ref_audio = _resample_audio(ref_audio, ref_sr, sr)
        min_len = min(len(audio), len(ref_audio))
        if min_len <= 0:
            continue
        audio = audio[:min_len]
        ref_audio = ref_audio[:min_len]

        channels = audio.shape[1]
        ref_channels = ref_audio.shape[1]
        out_channels = []
        for ch in range(channels):
            ref_ch = ref_audio[:, min(ch, ref_channels - 1)]
            f, _, z_in = stft(
                audio[:, ch],
                fs=sr,
                window="hann",
                nperseg=win_length,
                noverlap=win_length - hop_length,
                nfft=fft_size,
                boundary="zeros",
                padded=True,
            )
            _, _, z_ref = stft(
                ref_ch,
                fs=sr,
                window="hann",
                nperseg=win_length,
                noverlap=win_length - hop_length,
                nfft=fft_size,
                boundary="zeros",
                padded=True,
            )
            mag_in = np.abs(z_in)
            mag_ref = np.abs(z_ref)
            mask = mag_in / (mag_in + mag_ref + 1e-8)
            curve = _build_weight_curve(f, weights)
            mask = np.clip(mask * curve[:, None], 0.0, 1.0)

            if wiener_enabled:
                if np.var(mask) > 1e-12:
                    for _ in range(max(1, wiener_iterations)):
                        with np.errstate(divide="ignore", invalid="ignore"):
                            smoothed = wiener(mask, mysize=wiener_mysize)
                        smoothed = np.nan_to_num(smoothed, nan=0.0, posinf=1.0, neginf=0.0)
                        mask = (1.0 - wiener_strength) * mask + wiener_strength * smoothed
                    mask = np.clip(mask, 0.0, 1.0)

            if soft_enabled:
                strength = min(1.0, max(0.01, soft_strength))
                mask = np.clip(mask, 0.0, 1.0) ** strength

            z_out = z_in * mask
            _, out = istft(
                z_out,
                fs=sr,
                window="hann",
                nperseg=win_length,
                noverlap=win_length - hop_length,
                nfft=fft_size,
                input_onesided=True,
                boundary=True,
            )
            out_channels.append(out)

        out_audio = np.stack(out_channels, axis=1)
        out_audio = out_audio[:min_len]
        out_path = out_dir / f"{wav.stem} ({stage_cfg.get('name','spectral_mask')}).wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), out_audio, sr)
        outputs.append(out_path)
    return outputs


def run_midi_stage(stage_cfg: dict, out_dir: Path, inputs: List[Path]) -> List[Path]:
    """
    Convert stems to MIDI using basic-pitch and/or omnizart.
    stage_cfg options:
      engines: list of "basic_pitch" and/or "omnizart" (default both if available)
    """
    engines = stage_cfg.get("engines") or ["basic_pitch", "omnizart"]
    outputs: List[Path] = []

    if "basic_pitch" in engines:
        try:
            from basic_pitch.inference import predict
            has_bp = True
        except Exception:
            print("[warn] basic_pitch not available, skipping.")
            has_bp = False
    else:
        has_bp = False

    if "omnizart" in engines:
        try:
            import omnizart
            has_omni = True
        except Exception:
            print("[warn] omnizart not available, skipping.")
            has_omni = False
    else:
        has_omni = False

    for wav in inputs:
        if has_bp:
            try:
                midi_path = out_dir / f"{wav.stem} ({stage_cfg.get('name','midi')})_basic_pitch.mid"
                midi_path.parent.mkdir(parents=True, exist_ok=True)
                predict(str(wav), output_directory=str(midi_path.parent), midi=True, save_notes=False, visualize=False)
                # basic-pitch writes its own midi name; move it
                generated = midi_path.parent / f"{wav.stem}.mid"
                if generated.exists():
                    generated.rename(midi_path)
                outputs.append(midi_path)
            except Exception as exc:
                print(f"[error] basic_pitch failed on {wav.name}: {exc}")
        if has_omni:
            try:
                midi_path = out_dir / f"{wav.stem} ({stage_cfg.get('name','midi')})_omnizart.mid"
                midi_path.parent.mkdir(parents=True, exist_ok=True)
                omnizart.music.transcribe(str(wav), output=str(midi_path))
                outputs.append(midi_path)
            except Exception as exc:
                print(f"[error] omnizart failed on {wav.name}: {exc}")

    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Stem pipeline using audio-separator with YAML config.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("audio_file", help="Path to input audio file.")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    global_cfg = cfg.get("global_config") or {}
    stages = cfg.get("stages") or []

    mix_path = Path(args.audio_file).resolve()
    base_output = resolve_path(Path(), global_cfg.get("output_dir") or "./stems") or Path("./stems").resolve()
    ensure_dir(base_output)
    model_dir = resolve_path(config_path.parent, global_cfg.get("model_file_dir") or "") or Path()

    for stage in stages:
        run_stage(stage, global_cfg, mix_path, base_output, model_dir)

    print("Pipeline complete.")
    print(f"Outputs in: {base_output}")


if __name__ == "__main__":
    main()
