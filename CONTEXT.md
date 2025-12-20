# Project Context: Audio Source Separation Pipeline

## Goal
Build a robust, automated audio source separation pipeline using
- python-audio-separator
- UVR / MDX / MDXC / Demucs models

Primary objectives:
1. Extract highest-quality vocal stems (maximize SAR, then SDR)
2. Build ensemble vocals from multiple models
3. Create instrumental as mix − vocals ensemble
4. Post-process vocals (dereverb, denoise)
5. Post-process instrumentals (multi-stem, drums, guitars)

## Key Design Decisions
- Vocal quality prioritizes SAR over SDR
- Instrumental quality prioritizes SIR over SAR
- Ensemble merging uses weighted STFT magnitude
- Sample-rate mismatches are auto-resampled
- Model metadata comes from models.json (family → model → info)

## Key Files
- run_separation_pipeline.py
- download_models.py
- stem_ensemble.py
- separation_plan.yaml
- models.json (from audio-separator)

## Known Gotchas
- audio-separator JSON output is not flat:
  { "VR": { model_name: {...} }, "MDX": {...}, ... }
- model identifiers come from `filename` or `download_files`
- models output mixed sample rates (44.1k / 48k)
- NOT all models have published SDR/SIR/SAR scores

## Preferred Models (from models.json)
### Vocals (SAR → SDR)
1. vocals_mel_band_roformer.ckpt (SAR 13.44, SDR 12.6)
2. MDX23C-InstVoc_HQ.ckpt
3. htdemucs_ft.yaml

### Instrumentals (SIR → SAR)
1. MDX23C-InstVoc_HQ.ckpt
2. htdemucs_ft.yaml
3. htdemucs_6s.yaml

## Current State
- Scripts run end-to-end
- Ensemble vocals working
- Instrumental subtraction working
- Remaining improvements: automation, batch processing, CSV reporting
