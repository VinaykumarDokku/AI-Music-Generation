import os
import numpy as np
import pandas as pd
from tensorflow import keras
from music21 import stream, note, chord

notes = pd.read_csv("all_notes.csv", header=None)[0].astype(str).tolist()

vocabulary = sorted(set(notes))
note_to_int = {n: i for i, n in enumerate(vocabulary)}
int_to_note = {i: n for n, i in note_to_int.items()}

model = keras.models.load_model("models/best_model.keras")

sequence_length = 100
start = np.random.randint(0, len(notes) - sequence_length)

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
    index = int(np.argmax(prediction))

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
        new_element = note.Note(pattern)

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
