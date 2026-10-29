"""Tests for per-note velocity estimation.

The signals are synthesized here rather than mocked: the estimator's whole job
is to measure real energy in real audio, so a fake spectrogram would test
nothing.
"""

import numpy as np
import pytest

from muscriptor.tokenizer.notes import Note
from muscriptor_plus.velocity import (
    VelocityEstimator,
    VelocityOptions,
    pitch_to_hz,
)

SR = 16000


def tone(freq, dur, amp, decay=2.0):
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(1.0, t * 120) * np.exp(-t * decay)
    return (amp * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)


def note(onset, offset, pitch, is_drum=False):
    return Note(is_drum=is_drum, program=0, onset=onset, offset=offset, pitch=pitch)


def test_pitch_to_hz():
    assert pitch_to_hz(69) == pytest.approx(440.0)
    assert pitch_to_hz(60) == pytest.approx(261.6256, abs=1e-3)
    assert pitch_to_hz(81) == pytest.approx(880.0)


def test_louder_notes_get_higher_velocity():
    x = np.concatenate([tone(440, 0.4, 0.9), tone(440, 0.4, 0.3), tone(440, 0.4, 0.08)])
    notes = [note(0.0, 0.4, 69), note(0.4, 0.8, 69), note(0.8, 1.2, 69)]
    est = VelocityEstimator(VelocityOptions(auto_calibrate=False))
    v = est.estimate(x, notes)
    assert v[0] > v[1] > v[2]


def test_velocity_tracks_decibels_linearly_without_autocalibration():
    # A factor of 10 in amplitude is 20 dB; with a 34 dB span over 100 velocity
    # units that is ~59 units apart. Notes are separated by silence so the
    # measurement window cannot leak from one into the other.
    gap = np.zeros(int(SR * 0.25), dtype=np.float32)
    x = np.concatenate([tone(440, 0.4, 1.0), gap, tone(440, 0.4, 0.1)])
    notes = [note(0.0, 0.4, 69), note(0.65, 1.05, 69)]
    est = VelocityEstimator(VelocityOptions(auto_calibrate=False, center_db=-8.0, dynamic_range_db=34.0))
    v = est.estimate(x, notes)
    assert v[0] - v[1] == pytest.approx(20.0 / 34.0 * 100.0, abs=8)


def test_adjacent_notes_do_not_leak_into_each_other():
    # Back-to-back notes at very different amplitudes must still be told apart;
    # this is what the half-window measurement offset buys.
    x = np.concatenate([tone(440, 0.4, 1.0), tone(440, 0.4, 0.1)])
    notes = [note(0.0, 0.4, 69), note(0.4, 0.8, 69)]
    est = VelocityEstimator(VelocityOptions(auto_calibrate=False, center_db=-8.0, dynamic_range_db=34.0))
    v = est.estimate(x, notes)
    assert v[0] - v[1] > 40


def test_auto_calibration_uses_the_full_range():
    amps = [0.9, 0.6, 0.35, 0.2, 0.1, 0.05]
    x = np.concatenate([tone(440, 0.3, a) for a in amps])
    notes = [note(i * 0.3, i * 0.3 + 0.3, 69) for i in range(len(amps))]
    est = VelocityEstimator(VelocityOptions(auto_calibrate=True))
    v = est.estimate(x, notes)
    assert list(v) == sorted(v, reverse=True)
    assert v[0] >= 100
    assert v[-1] <= 50


def test_quieter_piece_yields_quieter_velocities():
    loud = np.concatenate([tone(440, 0.4, 0.8), tone(440, 0.4, 0.7)])
    quiet = np.concatenate([tone(440, 0.4, 0.08), tone(440, 0.4, 0.07)])
    notes = [note(0.0, 0.4, 69), note(0.4, 0.8, 69)]
    est = VelocityEstimator(VelocityOptions(auto_calibrate=False))
    v_loud = est.estimate(loud, notes)
    v_quiet = est.estimate(quiet, notes)
    assert v_loud.mean() > v_quiet.mean()


def test_pitch_is_measured_not_neighbouring_bins():
    # Two different pitches at the same loudness should land close together.
    x = np.concatenate([tone(pitch_to_hz(60), 0.4, 0.5), tone(pitch_to_hz(72), 0.4, 0.5)])
    notes = [note(0.0, 0.4, 60), note(0.4, 0.8, 72)]
    est = VelocityEstimator(VelocityOptions(auto_calibrate=False))
    v = est.estimate(x, notes)
    assert abs(int(v[0]) - int(v[1])) <= 8


def test_velocity_is_clamped_into_midi_range():
    # An enormous amplitude must not exceed the configured ceiling.
    x = np.concatenate([tone(440, 0.4, 5.0), tone(440, 0.4, 0.001)])
    notes = [note(0.0, 0.4, 69), note(0.4, 0.8, 69)]
    est = VelocityEstimator(VelocityOptions(auto_calibrate=False))
    v = est.estimate(x, notes)
    assert v.min() >= 1 and v.max() <= 127


def test_empty_notes_returns_empty():
    est = VelocityEstimator()
    assert est.estimate(np.zeros(SR, dtype=np.float32), []).shape == (0,)


def test_apply_sets_velocity_attribute():
    x = np.concatenate([tone(440, 0.4, 0.8), tone(440, 0.4, 0.1)])
    notes = [note(0.0, 0.4, 69), note(0.4, 0.8, 69)]
    est = VelocityEstimator(VelocityOptions(auto_calibrate=False))
    out = est.apply(x, notes)
    assert all(hasattr(n, "velocity") for n in out)
    assert out[0].velocity > out[1].velocity
    # originals are untouched
    assert not hasattr(notes[0], "velocity")


def test_drums_use_broadband_measurement():
    noise = np.random.default_rng(0).standard_normal(int(SR * 0.3)).astype(np.float32)
    x = np.concatenate([noise * 0.6, noise * 0.1])
    notes = [note(0.0, 0.3, 38, is_drum=True), note(0.3, 0.6, 38, is_drum=True)]
    est = VelocityEstimator(VelocityOptions(auto_calibrate=False))
    v = est.estimate(x, notes)
    assert v[0] > v[1]


def test_invalid_options_rejected():
    with pytest.raises(ValueError):
        VelocityOptions(min_velocity=0)
    with pytest.raises(ValueError):
        VelocityOptions(min_velocity=100, max_velocity=50)
    with pytest.raises(ValueError):
        VelocityOptions(dynamic_range_db=0)
    with pytest.raises(ValueError):
        VelocityOptions(calibration_percentiles=(90, 10))
