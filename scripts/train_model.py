import os
import numpy as np
import pandas as pd
from tensorflow import keras

# Load MIDI note data
notes = pd.read_csv("all_notes.csv", header=None)[0].astype(str).tolist()

sequence_length = 100
vocabulary = sorted(set(notes))
note_to_int = {note: i for i, note in enumerate(vocabulary)}

X, y = [], []

for i in range(len(notes) - sequence_length):
    X.append([note_to_int[n] for n in notes[i:i + sequence_length]])
    y.append(note_to_int[notes[i + sequence_length]])

X = np.array(X, dtype=np.float32).reshape(-1, sequence_length, 1)
X /= len(vocabulary)
y = np.array(y, dtype=np.int32)

print("Training samples:", len(X))
print("Vocabulary size:", len(vocabulary))

model = keras.Sequential([
    keras.Input(shape=(sequence_length, 1)),
    keras.layers.LSTM(128),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(len(vocabulary), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

os.makedirs("models", exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "models/best_model.keras",
        monitor="val_loss",
        save_best_only=True
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

model.fit(
    X,
    y,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks
)

model.save("models/final_model.keras")
print("Training completed. Model saved!")
