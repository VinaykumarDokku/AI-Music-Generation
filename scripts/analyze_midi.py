from pathlib import Path
import argparse
import pretty_midi


def analyze_midi(path):
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = [
        note
        for instrument in midi.instruments
        for note in instrument.notes
    ]

    print(f"\nFile: {path.name}")
    print(f"Duration: {midi.get_end_time():.2f} seconds")
    print(f"Instruments: {len(midi.instruments)}")
    print(f"Notes: {len(notes)}")

    if notes:
        avg_velocity = sum(n.velocity for n in notes) / len(notes)
        print(f"Average velocity: {avg_velocity:.1f}")
        print(f"Pitch range: {min(n.pitch for n in notes)}-"
              f"{max(n.pitch for n in notes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a MIDI music file"
    )
    parser.add_argument("file", type=Path)
    args = parser.parse_args()

    if not args.file.is_file():
        parser.error("MIDI file does not exist")

    analyze_midi(args.file)
