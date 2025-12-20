#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List
import shutil

import yaml

from audio_separator.separator import Separator
from stem_ensemble import build_vocals_ensemble, build_instrumental_ensemble


VOCAL_KEYWORDS = ["vocal", "vox", "sing", "voice"]
INSTRUMENTAL_KEYWORDS = ["instrumental", "inst", "music", "backing", "accompaniment"]


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


def run_stage(stage_cfg: dict, global_cfg: dict, mix_path: Path, base_output: Path) -> List[Path]:
    stage_type = stage_cfg.get("type", "separate").lower()
    inputs = collect_inputs(stage_cfg, mix_path, base_output)
    if not inputs:
        print(f"[{stage_cfg.get('name','?')}] No inputs, skipping.")
        return []

    out_dir = ensure_dir(resolve_path(base_output, stage_cfg.get("output_dir") or "."))

    if stage_type == "ensemble":
        return run_ensemble_stage(stage_cfg, out_dir, inputs)

    # separate
    all_outputs: List[Path] = []
    model_dir = resolve_path(Path(), global_cfg.get("model_file_dir") or "") or Path()
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
    return all_outputs


def run_ensemble_stage(stage_cfg: dict, out_dir: Path, inputs: List[Path]) -> List[Path]:
    mode = stage_cfg.get("mode", "vocals").lower()
    top_k = stage_cfg.get("top_k")
    out_path = out_dir / (stage_cfg.get("output_name") or f"{mode}_ensemble.wav")
    if mode == "vocals":
        candidates = [p for p in inputs if any(k in p.name.lower() for k in VOCAL_KEYWORDS)]
        scores = {p: 0.0 for p in candidates}
        build_vocals_ensemble(candidates, scores, out_path, top_k=min(top_k or len(candidates), len(candidates)))
        print(f"[{stage_cfg.get('name','?')}] Vocals ensemble -> {out_path}")
        return [out_path]
    if mode == "instrumental":
        candidates = [p for p in inputs if any(k in p.name.lower() for k in INSTRUMENTAL_KEYWORDS)]
        build_instrumental_ensemble(candidates, out_path, vocals_path=None)
        print(f"[{stage_cfg.get('name','?')}] Instrumental ensemble -> {out_path}")
        return [out_path]
    print(f"[{stage_cfg.get('name','?')}] Unknown ensemble mode '{mode}', skipping.")
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Stem pipeline using audio-separator with YAML config.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("audio_file", help="Path to input audio file.")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    global_cfg = cfg.get("global_config") or {}
    stages = cfg.get("stages") or []

    mix_path = Path(args.audio_file).resolve()
    base_output = resolve_path(Path(), global_cfg.get("output_dir") or "./stems") or Path("./stems").resolve()
    ensure_dir(base_output)

    for stage in stages:
        run_stage(stage, global_cfg, mix_path, base_output)

    print("Pipeline complete.")
    print(f"Outputs in: {base_output}")


if __name__ == "__main__":
    main()
