"""Tests for phrase shaping and micro-timing."""

import pytest

from muscriptor_plus.humanize import (
    HumanizeOptions,
    Humanizer,
    _deterministic_unit,
    _group_phrases,
    _phrase_arc,
)
from muscriptor_plus.midi_writer import VelNote


def note(onset, pitch=60, velocity=100, offset=None):
    return VelNote(
        is_drum=False,
        program=0,
        onset=onset,
        offset=offset if offset is not None else onset + 0.3,
        pitch=pitch,
        velocity=velocity,
    )


def test_phrase_arc_shapes():
    arch = _phrase_arc(5, "arch")
    assert arch[0] == pytest.approx(0.0)
    assert arch[2] == pytest.approx(1.0)
    assert arch[-1] == pytest.approx(0.0)
    cres = _phrase_arc(5, "crescendo")
    assert cres[0] < cres[-1]
    dim = _phrase_arc(5, "diminuendo")
    assert dim[0] > dim[-1]
    assert _phrase_arc(5, "none") == [0.0] * 5


def test_group_phrases_splits_on_silence():
    notes = [
        note(0.0), note(0.2), note(0.4),  # phrase 1
        note(2.0), note(2.2),            # phrase 2 (gap 1.6s)
    ]
    phrases = _group_phrases(notes, gap_s=0.45)
    assert [sorted(p) for p in phrases] == [[0, 1, 2], [3, 4]]


def test_phrase_shaping_creates_an_arch():
    notes = [note(i * 0.3, velocity=90) for i in range(7)]
    h = Humanizer(HumanizeOptions(phrase_depth=12.0, phrase_shape="arch", timing_jitter_s=0.0, accent_beat=0.0, accent_downbeat=0.0))
    out = h.humanize(notes)
    vels = [n.velocity for n in out]
    assert vels[3] > vels[0]
    assert vels[3] > vels[-1]


def test_phrase_shaping_preserves_mean_loudness():
    notes = [note(i * 0.3, velocity=90) for i in range(9)]
    h = Humanizer(HumanizeOptions(phrase_depth=12.0, timing_jitter_s=0.0, accent_beat=0.0, accent_downbeat=0.0))
    out = h.humanize(notes)
    before = sum(n.velocity for n in notes) / len(notes)
    after = sum(n.velocity for n in out) / len(out)
    assert abs(before - after) <= 3


def test_timing_jitter_is_deterministic():
    notes = [note(i * 0.3) for i in range(5)]
    opts = HumanizeOptions(timing_jitter_s=0.02, seed=42, phrase_depth=0.0, accent_beat=0.0, accent_downbeat=0.0)
    a = [n.onset for n in Humanizer(opts).humanize(notes)]
    b = [n.onset for n in Humanizer(opts).humanize(notes)]
    assert a == b


def test_timing_jitter_bounded():
    notes = [note(i * 0.3) for i in range(20)]
    opts = HumanizeOptions(timing_jitter_s=0.012, seed=1, phrase_depth=0.0, accent_beat=0.0, accent_downbeat=0.0)
    out = Humanizer(opts).humanize(notes)
    for original, humanized in zip(notes, out):
        assert abs(humanized.onset - original.onset) <= 0.012 + 1e-9


def test_accents_boost_downbeats():
    # At 120 BPM the beat is 0.5 s; a 4/4 bar has a downbeat every 2 s.
    notes = [note(i * 0.5, velocity=80) for i in range(5)]
    opts = HumanizeOptions(
        phrase_depth=0.0, timing_jitter_s=0.0, bpm=120.0, beats_per_bar=4,
        accent_beat=5.0, accent_downbeat=9.0,
    )
    out = Humanizer(opts).humanize(notes)
    assert out[0].velocity > out[1].velocity  # downbeat vs beat
    assert out[4].velocity > out[1].velocity  # next downbeat


def test_velocities_stay_in_midi_range():
    notes = [note(i * 0.3, velocity=127) for i in range(6)]
    opts = HumanizeOptions(phrase_depth=40.0, timing_jitter_s=0.0)
    out = Humanizer(opts).humanize(notes)
    assert all(1 <= n.velocity <= 127 for n in out)


def test_offsets_are_kept_positive():
    notes = [note(0.0, offset=0.05), note(0.3, offset=0.35)]
    opts = HumanizeOptions(timing_jitter_s=0.05, seed=7)
    out = Humanizer(opts).humanize(notes)
    assert all(n.offset > n.onset for n in out)


def test_zero_options_is_identity():
    notes = [note(i * 0.3) for i in range(4)]
    opts = HumanizeOptions(phrase_depth=0.0, timing_jitter_s=0.0, accent_beat=0.0, accent_downbeat=0.0)
    out = Humanizer(opts).humanize(notes)
    assert [n.onset for n in out] == [n.onset for n in notes]
    assert [n.velocity for n in out] == [n.velocity for n in notes]


def test_invalid_options_rejected():
    with pytest.raises(ValueError):
        HumanizeOptions(phrase_shape="wobble")
    with pytest.raises(ValueError):
        HumanizeOptions(timing_jitter_s=-1)


def test_deterministic_unit_range():
    for i in range(100):
        u = _deterministic_unit("seed", i)
        assert -1.0 <= u <= 1.0
