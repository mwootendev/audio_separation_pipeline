#!/usr/bin/env python3
"""
csv_chords_to_midi_strums.py

Convert a chord CSV file of the form:

    chord,start,end
    G:min,0.185759636,24.009432953
    A#:maj,24.055872862,29.953741305
    D#:maj,30.000181214,32.97233539
    F:maj,33.018775299,35.898049657

into a MIDI file containing guitar- or piano-style chord events.

Behavior:
    - If NO audio reference is provided:
        One sustained chord event per CSV row.
    - If an audio reference is provided via --audio:
        Uses librosa onset detection to find individual "strums" and
        creates chord hits at each detected onset time that falls within
        a chord's CSV interval.

Usage:
    python csv_to_chords_midi_strums.py input.csv output.mid \
        [--bpm 120] [--voicing guitar|piano] [--audio reference.wav]

Defaults:
    bpm = 120
    voicing = "guitar"

Requires:
    pip install MIDIUtil librosa soundfile
"""

import argparse
import csv
from typing import List, Tuple, Optional

from midiutil import MIDIFile

# --- Music theory helpers ----------------------------------------------------

NOTE_NAME_TO_PC = {
    "C": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11,
}

CHORD_QUALITIES = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "7":    [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
}


def parse_chord_symbol(symbol: str) -> Tuple[int, List[int]]:
    """
    Parse a chord symbol like "G:min" or "A#:maj7" into (root_pc, intervals).

    Returns:
        root_pc: pitch class (0=C, 1=C#, ..., 11=B)
        intervals: list of semitone offsets from root
    """
    symbol = symbol.strip()
    if ":" in symbol:
        root_str, quality_str = symbol.split(":", 1)
    else:
        # If quality omitted, assume major
        root_str, quality_str = symbol, "maj"

    root_str = root_str.strip()
    quality_str = quality_str.strip()

    if root_str not in NOTE_NAME_TO_PC:
        raise ValueError(f"Unsupported root note in chord symbol: {symbol}")

    root_pc = NOTE_NAME_TO_PC[root_str]

    # Normalize a few common aliases
    quality_str = quality_str.lower()
    if quality_str in ("m", "min", "minor"):
        quality_key = "min"
    elif quality_str in ("maj", "major", ""):
        quality_key = "maj"
    elif quality_str in ("dim", "diminished"):
        quality_key = "dim"
    elif quality_str in ("aug", "augmented"):
        quality_key = "aug"
    elif quality_str in ("7", "dom7", "dom"):
        quality_key = "7"
    elif quality_str in ("maj7", "ma7", "Δ7"):
        quality_key = "maj7"
    elif quality_str in ("min7", "m7"):
        quality_key = "min7"
    else:
        raise ValueError(f"Unsupported chord quality in symbol: {symbol}")

    intervals = CHORD_QUALITIES[quality_key]
    return root_pc, intervals


def chord_to_midi_notes(root_pc: int, intervals: List[int], voicing: str) -> List[int]:
    """
    Map a chord defined by root pitch class and intervals to concrete MIDI note numbers.

    voicing:
        "guitar" - root around lower-mid register, roughly like open/barre chords.
        "piano"  - root in mid register.
    """
    if voicing == "piano":
        base_octave = 4  # C4 = 60
    else:
        # Guitar-ish register: roots around C3–E3
        base_octave = 3

    root_midi = root_pc + 12 * base_octave
    notes = [root_midi + interval for interval in intervals]

    return notes


def seconds_to_beats(t_seconds: float, bpm: float) -> float:
    """Convert seconds to quarter-note beats given BPM."""
    return t_seconds * bpm / 60.0


# --- CSV / audio helpers -----------------------------------------------------

def read_chord_csv(path: str) -> List[Tuple[str, float, float]]:
    """
    Read a chord CSV of the form chord,start,end.

    Returns list of tuples: (chord_symbol, start_seconds, end_seconds)
    """
    rows: List[Tuple[str, float, float]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header_seen = False
        for row in reader:
            if not row:
                continue
            # Detect header row if present
            if not header_seen and row[0].lower().startswith("chord"):
                header_seen = True
                continue
            if len(row) < 3:
                continue
            chord_symbol = row[0].strip()
            start = float(row[1])
            end = float(row[2])
            rows.append((chord_symbol, start, end))
    return rows


def detect_onsets_with_librosa(audio_path: str) -> List[float]:
    """
    Use librosa to detect onset (transient) times in the audio.

    Returns:
        List of onset times in seconds (floats).
    """
    import librosa

    # Load audio in mono at native sample rate
    y, sr = librosa.load(audio_path, mono=True)

    # Use librosa's onset detection (frames)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")

    # Convert frames to time in seconds
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    return list(onset_times)


# --- MIDI creation -----------------------------------------------------------

def create_midi_from_chords(
    chords: List[Tuple[str, float, float]],
    bpm: float,
    voicing: str,
    output_path: str,
    onset_times: Optional[List[float]] = None,
) -> None:
    """
    Create a MIDI file from chord data.

    chords:
        list of (chord_symbol, start_seconds, end_seconds)
    bpm:
        tempo in beats per minute
    voicing:
        "guitar" or "piano"
    output_path:
        output .mid filename
    onset_times:
        If provided, these are onset times (seconds) from the reference audio.
        Chord hits are created at each onset that falls inside a chord interval.
        If None, one sustained chord is created per CSV row.
    """
    voicing = voicing.lower()
    if voicing not in ("guitar", "piano"):
        raise ValueError("voicing must be 'guitar' or 'piano'")

    # MIDI setup
    num_tracks = 1
    track = 0
    channel = 0

    mf = MIDIFile(numTracks=num_tracks, adjust_origin=True)
    mf.addTrackName(track, time=0, trackName=f"{voicing.capitalize()} Chords")
    mf.addTempo(track, time=0, tempo=bpm)

    # Choose instrument program by voicing (General MIDI)
    if voicing == "piano":
        program = 0   # Acoustic Grand Piano (0-based)
    else:
        program = 24  # Nylon Guitar (0-based)

    mf.addProgramChange(track, channel, time=0, program=program)

    volume = 90

    if onset_times is None or len(onset_times) == 0:
        # --- Original behavior: one sustained block chord per CSV row ----
        for chord_symbol, start_sec, end_sec in chords:
            try:
                root_pc, intervals = parse_chord_symbol(chord_symbol)
            except ValueError as e:
                print(f"Warning: {e}; skipping chord '{chord_symbol}'")
                continue

            notes = chord_to_midi_notes(root_pc, intervals, voicing)
            start_beats = seconds_to_beats(start_sec, bpm)
            end_beats = seconds_to_beats(end_sec, bpm)
            duration_beats = max(0.0, end_beats - start_beats)

            if duration_beats <= 0.0:
                print(f"Warning: non-positive duration for chord '{chord_symbol}', skipping.")
                continue

            for pitch in notes:
                mf.addNote(
                    track=track,
                    channel=channel,
                    pitch=pitch,
                    time=start_beats,
                    duration=duration_beats,
                    volume=volume,
                )
    else:
        # --- Strum-based behavior: hit chords at each onset ----------------
        onset_times_sorted = sorted(onset_times)

        for chord_symbol, start_sec, end_sec in chords:
            try:
                root_pc, intervals = parse_chord_symbol(chord_symbol)
            except ValueError as e:
                print(f"Warning: {e}; skipping chord '{chord_symbol}'")
                continue

            notes = chord_to_midi_notes(root_pc, intervals, voicing)

            # Collect onset indices that fall into this chord's time range
            onset_indices = [
                idx for idx, t in enumerate(onset_times_sorted)
                if start_sec <= t < end_sec
            ]

            if not onset_indices:
                # No onset found within this chord; fallback to a sustained block chord
                start_beats = seconds_to_beats(start_sec, bpm)
                end_beats = seconds_to_beats(end_sec, bpm)
                duration_beats = max(0.0, end_beats - start_beats)
                if duration_beats <= 0.0:
                    continue
                for pitch in notes:
                    mf.addNote(
                        track=track,
                        channel=channel,
                        pitch=pitch,
                        time=start_beats,
                        duration=duration_beats,
                        volume=volume,
                    )
                continue

            for idx in onset_indices:
                t_onset = onset_times_sorted[idx]

                # Determine an appropriate end time for this strum:
                # Use either the next onset time or the end of the chord,
                # whichever comes first.
                if idx < len(onset_times_sorted) - 1:
                    t_next_onset = onset_times_sorted[idx + 1]
                    t_end_strum = min(t_next_onset, end_sec)
                else:
                    t_end_strum = end_sec

                if t_end_strum <= t_onset:
                    # Fallback: give a small fixed duration in seconds
                    duration_sec = 0.1
                else:
                    duration_sec = t_end_strum - t_onset

                start_beats = seconds_to_beats(t_onset, bpm)
                duration_beats = seconds_to_beats(duration_sec, bpm)

                for pitch in notes:
                    mf.addNote(
                        track=track,
                        channel=channel,
                        pitch=pitch,
                        time=start_beats,
                        duration=duration_beats,
                        volume=volume,
                    )

    with open(output_path, "wb") as out_f:
        mf.writeFile(out_f)


# --- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a chord CSV file into a MIDI file with guitar or piano "
            "voicings, optionally using a reference audio file to infer "
            "individual strum/onset times via librosa."
        )
    )
    parser.add_argument("input_csv", help="Path to input CSV file (chord,start,end).")
    parser.add_argument("output_midi", help="Path to output MIDI file.")
    parser.add_argument(
        "--bpm",
        type=float,
        default=120.0,
        help="Tempo in beats per minute (default: 120).",
    )
    parser.add_argument(
        "--voicing",
        choices=["guitar", "piano"],
        default="guitar",
        help="Chord voicing style: guitar or piano (default: guitar).",
    )
    parser.add_argument(
        "--audio",
        help=(
            "Optional reference audio file (e.g. WAV) for onset/strum detection. "
            "If provided, librosa is used to detect onsets, and chord hits are "
            "placed at those times within each chord interval."
        ),
        default=None,
    )

    args = parser.parse_args()

    chords = read_chord_csv(args.input_csv)
    if not chords:
        print("No chord rows found in CSV. Exiting.")
        return

    onset_times = None
    if args.audio is not None:
        print(f"Detecting onsets in audio: {args.audio}")
        onset_times = detect_onsets_with_librosa(args.audio)
        print(f"Detected {len(onset_times)} onsets.")

    create_midi_from_chords(
        chords=chords,
        bpm=args.bpm,
        voicing=args.voicing,
        output_path=args.output_midi,
        onset_times=onset_times,
    )
    print(f"Written MIDI file to {args.output_midi}")


if __name__ == "__main__":
    main()
