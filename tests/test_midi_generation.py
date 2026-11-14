import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from music21 import converter, note, chord


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_music.py"
OUTPUT = ROOT / "generated_music" / "ai_generated_song.mid"


def generate_music(temperature, seed):
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--temperature",
            str(temperature),
            "--seed",
            str(seed),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"Music generation failed:\n{result.stderr}"
    )

    assert OUTPUT.exists(), "MIDI output file was not created"

    return OUTPUT.read_bytes()


def get_notes(midi_path):
    score = converter.parse(str(midi_path))
    return list(score.flatten().notes)


def test_midi_file_exists_and_is_valid(tmp_path):
    midi_bytes = generate_music(0.7, 42)

    path = tmp_path / "test.mid"
    path.write_bytes(midi_bytes)

    events = get_notes(path)

    assert len(events) > 0
    assert all(
        isinstance(event, (note.Note, chord.Chord))
        for event in events
    )


def test_temperature_sampling_reproducibility():
    first = generate_music(0.7, 42)
    second = generate_music(0.7, 42)

    assert first == second, (
        "Same temperature and seed produced different MIDI files"
    )


def test_different_temperatures_can_change_output():
    low = generate_music(0.3, 42)
    high = generate_music(1.5, 42)

    assert low != high, (
        "Different temperatures produced identical MIDI files"
    )


def test_invalid_temperature_is_rejected():
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--temperature",
            "0",
            "--seed",
            "42",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0, (
        "Temperature zero should be rejected"
    )
