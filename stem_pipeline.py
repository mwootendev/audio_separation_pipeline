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



def _resample_inputs(inputs: List[Path], target_sr: int, work_dir: Path) -> List[Path]:
    import numpy as np
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    work_dir.mkdir(parents=True, exist_ok=True)
    resampled: List[Path] = []
    for idx, wav in enumerate(inputs):
        audio, sr = sf.read(str(wav), always_2d=True)
        if sr == target_sr:
            resampled.append(wav)
            continue
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        y = resample_poly(audio, up, down, axis=0).astype(np.float32)
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


def rename_outputs(generated: List[Path], target_dir: Path, input_stem: str, model_name: str, output_names: dict | None) -> List[Path]:
    renamed: List[Path] = []
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
    if stage_type in ["ensemble", "filter"]:
        target_sr = global_cfg.get("sample_rate")
        if target_sr:
            resample_dir = out_dir / "_work" / "resample"
            inputs_for_stage = _resample_inputs(inputs, int(target_sr), resample_dir)

    outputs: List[Path] = []
    if stage_type == "ensemble":
        outputs = run_ensemble_stage(stage_cfg, out_dir, inputs_for_stage)
    elif stage_type == "filter":
        outputs = run_filter_stage(stage_cfg, out_dir, inputs_for_stage)
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
                renamed = rename_outputs(generated_paths, out_dir, audio_file.stem, friendly, output_names)
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
            renamed = rename_outputs(downloaded, out_dir, audio_file.stem, friendly, output_names)
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
    combine = stage_cfg.get("combine_type") or stage_cfg.get("combine")
    out_path = out_dir / (stage_cfg.get("output_name") or f"{mode}_ensemble.wav")
    if mode == "vocals":
        candidates = [p for p in inputs if any(k in p.name.lower() for k in VOCAL_KEYWORDS)]
        scores = {p: 0.0 for p in candidates}
        build_vocals_ensemble(candidates, scores, out_path, top_k=min(top_k or len(candidates), len(candidates)), combine=(combine or "avg"))
        print(f"[{stage_cfg.get('name','?')}] Vocals ensemble -> {out_path}")
        return [out_path]
    if mode == "instrumental":
        candidates = [p for p in inputs if any(k in p.name.lower() for k in INSTRUMENTAL_KEYWORDS)]
        combine_use = (combine or "median").lower()
        build_instrumental_ensemble(candidates, out_path, vocals_path=None, combine=combine_use)
        print(f"[{stage_cfg.get('name','?')}] Instrumental ensemble -> {out_path}")
        return [out_path]
    print(f"[{stage_cfg.get('name','?')}] Unknown ensemble mode '{mode}', skipping.")
    return []


def run_filter_stage(stage_cfg: dict, out_dir: Path, inputs: List[Path]) -> List[Path]:
    """
    Apply keep-bands filters to inputs.
    stage_cfg expects:
      bands: list of [lo, hi] Hz ranges to keep (summed)
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import butter, sosfilt

    bands = stage_cfg.get("bands") or []
    if not bands:
        print(f"[{stage_cfg.get('name','?')}] No bands specified, skipping.")
        return []

    outputs: List[Path] = []
    for wav in inputs:
        audio, sr = sf.read(str(wav), always_2d=True)
        x = audio.astype(np.float32)
        y = np.zeros_like(x)
        for lo, hi in bands:
            lo = max(10.0, float(lo))
            hi = float(hi)
            hi = min(0.49 * sr, hi)
            sos = butter(6, [lo, hi], btype="bandpass", fs=sr, output="sos")
            # Filter along the time axis (axis=0) per channel; avoids cross-channel bleed.
            y += sosfilt(sos, x, axis=0)
        # Normalize lightly
        peak = np.max(np.abs(y)) + 1e-12
        y = y * (0.99 / peak)
        out_path = out_dir / f"{wav.stem} ({stage_cfg.get('name','filter')}).wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), y, sr)
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
