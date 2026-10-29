"""End-to-end test of the enhanced pipeline against the real model.

This is marked ``slow`` and skipped unless the weights are present, because a
CPU decode of even a few seconds of audio takes tens of seconds. Run it with::

    pytest -m slow tests/test_enhanced_e2e.py

It is the only test that exercises the full path - model load, transcription,
velocity estimation, humanizing and MIDI writing - on real model output.
"""

import io
from pathlib import Path

import numpy as np
import pytest

from muscriptor_plus.enhanced import EnhancedTranscriber
from muscriptor_plus.humanize import HumanizeOptions
from muscriptor_plus.velocity import VelocityOptions

WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "muscriptor-large" / "model.safetensors"

pytestmark = pytest.mark.slow

SR = 16000


def tone(freq, dur, amp):
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(1, t * 60) * np.exp(-t * 1.8)
    return (amp * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)


@pytest.fixture(scope="module")
def model():
    if not WEIGHTS.exists():
        pytest.skip(f"weights not found at {WEIGHTS}")
    from muscriptor.transcription_model import TranscriptionModel

    return TranscriptionModel.load_model(weights_path=str(WEIGHTS), device="cpu", dtype="float32")


def test_loud_then_quiet_phrase_gets_distinct_velocities(model):
    loud = np.concatenate([tone(261.63, 0.45, 0.95), tone(329.63, 0.45, 0.95), tone(392.0, 0.5, 0.95)])
    quiet = np.concatenate([tone(261.63, 0.45, 0.12), tone(329.63, 0.45, 0.12), tone(392.0, 0.5, 0.12)])
    gap = np.zeros(int(SR * 0.4), dtype=np.float32)
    x = np.concatenate([loud, gap, quiet])

    import torch

    enh = EnhancedTranscriber(model, velocity=VelocityOptions(auto_calibrate=False))
    result = enh.transcribe((torch.from_numpy(x)[None, :], SR))

    assert len(result.notes) > 0
    loud_vels = [v for n, v in zip(result.notes, result.velocities) if n.onset < 2.0]
    quiet_vels = [v for n, v in zip(result.notes, result.velocities) if n.onset >= 2.0]
    assert loud_vels and quiet_vels
    assert np.mean(loud_vels) > np.mean(quiet_vels) + 15

    # The MIDI must actually carry those velocities.
    from mido import MidiFile

    midi = MidiFile(file=io.BytesIO(result.midi))
    velocities = {m.velocity for tr in midi.tracks for m in tr if m.type == "note_on"}
    assert len(velocities) > 1


def test_humanize_changes_output_but_not_pitch(model):
    x = np.concatenate([tone(261.63, 0.4, 0.8), tone(329.63, 0.4, 0.8), tone(392.0, 0.4, 0.8)])

    import torch

    enh = EnhancedTranscriber(
        model,
        velocity=VelocityOptions(auto_calibrate=False),
        humanize=HumanizeOptions(phrase_depth=0.0, timing_jitter_s=0.0, accent_beat=0.0, accent_downbeat=0.0),
    )
    flat = enh.transcribe((torch.from_numpy(x)[None, :], SR))

    enh2 = EnhancedTranscriber(
        model,
        velocity=VelocityOptions(auto_calibrate=False),
        humanize=HumanizeOptions(phrase_depth=15.0, timing_jitter_s=0.01, seed=3),
    )
    shaped = enh2.transcribe((torch.from_numpy(x)[None, :], SR))

    assert sorted(n.pitch for n in flat.notes) == sorted(n.pitch for n in shaped.notes)
    assert flat.midi != shaped.midi
