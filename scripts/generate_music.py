
def convert_pattern_to_music21(pattern, offset):
    """Safely convert a note or chord token into music21."""
    from music21 import note, chord, pitch

    pattern = str(pattern).strip()

    if not pattern:
        return None

    try:
        if "+" in pattern or "." in pattern:
            parts = pattern.replace("+", ".").split(".")
            pitches = []

            for part in parts:
                part = part.strip()

                if not part:
                    continue

                try:
                    if part.lstrip("-").isdigit():
                        pitches.append(int(part))
                    else:
                        pitches.append(pitch.Pitch(part))
                except (ValueError, pitch.PitchException):
                    print("Skipping invalid chord component:", part)

            if not pitches:
                return None

            element = chord.Chord(pitches)
        else:
            if pattern.lstrip("-").isdigit():
                element = note.Note(int(pattern))
            else:
                element = note.Note(pattern)

        element.offset = offset
        element.quarterLength = 0.5
        return element

    except (ValueError, TypeError, pitch.PitchException) as error:
        print("Skipping invalid musical pattern:", pattern, error)
        return None




import os
import numpy as np
import argparse

import pandas as pd
from tensorflow import keras
from music21 import stream, note, chord


def sample_with_temperature(prediction, temperature, rng):
    probabilities = np.asarray(prediction, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, None)
    probabilities /= probabilities.sum()

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero")

    logits = np.log(probabilities) / temperature
    logits -= np.max(logits)

    adjusted = np.exp(logits)
    adjusted /= adjusted.sum()

    return int(rng.choice(len(adjusted), p=adjusted))



parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

temperature = args.temperature
rng = np.random.default_rng(args.seed)

notes = pd.read_csv("all_notes.csv", header=None)[0].astype(str).tolist()

vocabulary = sorted(set(notes))
note_to_int = {n: i for i, n in enumerate(vocabulary)}
int_to_note = {i: n for n, i in note_to_int.items()}

model = keras.models.load_model("models/best_model.keras")

sequence_length = 100
start = rng.integers(0, len(notes) - sequence_length)

pattern = [
    note_to_int[n]
    for n in notes[start:start + sequence_length]
]

generated = []

for i in range(200):
    x = np.array(pattern, dtype=np.float32).reshape(
        1, sequence_length, 1
    )
    x /= len(vocabulary)

    prediction = model.predict(x, verbose=0)[0]
    index = sample_with_temperature(prediction, temperature, rng)

    generated.append(int_to_note[index])
    pattern = pattern[1:] + [index]

    if (i + 1) % 50 == 0:
        print(f"Generated {i + 1}/200 notes")

output_notes = []
offset = 0.0

for pattern in generated:
    if "." in pattern or pattern.isdigit():
        pitches = pattern.split(".")
        notes_in_chord = [
            note.Note(int(pitch)) for pitch in pitches
        ]
        new_element = chord.Chord(notes_in_chord)
    else:
        new_element = convert_pattern_to_music21(pattern, offset)

    new_element.offset = offset
    new_element.quarterLength = 0.5
    output_notes.append(new_element)
    offset += 0.5

midi_stream = stream.Stream(output_notes)

os.makedirs("generated_music", exist_ok=True)
output_path = "generated_music/ai_generated_song.mid"

midi_stream.write("midi", fp=output_path)

print("Music generated successfully!")
print("Saved:", output_path)
