"""Configuration validation for AI music generation."""

from dataclasses import dataclass


@dataclass
class MusicConfig:
    tempo: int = 120
    duration_seconds: int = 30
    temperature: float = 1.0
    sequence_length: int = 100


def validate_config(config: MusicConfig) -> None:
    """Validate generation parameters."""
    if not 30 <= config.tempo <= 240:
        raise ValueError("Tempo must be between 30 and 240 BPM")

    if not 1 <= config.duration_seconds <= 600:
        raise ValueError("Duration must be between 1 and 600 seconds")

    if not 0.1 <= config.temperature <= 2.0:
        raise ValueError("Temperature must be between 0.1 and 2.0")

    if not 1 <= config.sequence_length <= 1024:
        raise ValueError("Sequence length must be between 1 and 1024")


if __name__ == "__main__":
    config = MusicConfig(
        tempo=120,
        duration_seconds=30,
        temperature=0.8,
        sequence_length=100,
    )

    validate_config(config)
    print("Music generation configuration is valid")
    print(config)
