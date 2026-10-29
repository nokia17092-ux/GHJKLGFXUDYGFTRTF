"""Tests for artifact cleanup."""

import numpy as np

from muscriptor_plus.cleanup import CleanupOptions, cleanup
from muscriptor_plus.midi_writer import VelNote


def note(onset, pitch, offset=None, program=0, is_drum=False, velocity=100):
    return VelNote(
        is_drum=is_drum,
        program=program,
        onset=onset,
        offset=offset if offset is not None else onset + 0.4,
        pitch=pitch,
        velocity=velocity,
    )


def test_drops_very_short_notes():
    notes = [note(0.0, 60), note(1.0, 62, offset=1.01)]  # 10 ms blip
    out = cleanup(notes, options=CleanupOptions(drop_restrike_overlaps=False, merge_gap_s=0.0))
    assert [n.pitch for n in out] == [60]


def test_keeps_drum_hits_even_if_short():
    notes = [note(0.0, 38, offset=0.001, is_drum=True)]
    out = cleanup(notes, options=CleanupOptions(merge_gap_s=0.0))
    assert len(out) == 1


def test_drops_restrike_inside_sounding_note():
    notes = [note(0.0, 60, offset=1.0), note(0.5, 60, offset=0.6)]
    out = cleanup(notes, options=CleanupOptions(merge_gap_s=0.0, min_duration_s=0.0))
    assert len(out) == 1
    assert out[0].offset == 1.0


def test_merges_blips_of_same_pitch():
    notes = [note(0.0, 60, offset=0.3), note(0.32, 60, offset=0.6)]
    out = cleanup(notes, options=CleanupOptions(drop_restrike_overlaps=False, min_duration_s=0.0, merge_gap_s=0.05))
    assert len(out) == 1
    assert out[0].offset == 0.6


def test_does_not_merge_distant_notes():
    notes = [note(0.0, 60, offset=0.3), note(1.0, 60, offset=1.3)]
    out = cleanup(notes, options=CleanupOptions(drop_restrike_overlaps=False, min_duration_s=0.0, merge_gap_s=0.05))
    assert len(out) == 2


def test_different_pitches_not_merged():
    notes = [note(0.0, 60, offset=0.3), note(0.32, 64, offset=0.6)]
    out = cleanup(notes, options=CleanupOptions(drop_restrike_overlaps=False, min_duration_s=0.0, merge_gap_s=0.05))
    assert len(out) == 2


def test_max_removed_fraction_guard():
    # 5 of 6 notes are blips; the guard should refuse to delete 83% and return
    # the input unchanged rather than wiping the transcription out.
    notes = [note(0.0, 60)] + [
        note(1.0 + i * 0.1, 61 + i, offset=1.0 + i * 0.1 + 0.01) for i in range(5)
    ]
    out = cleanup(notes, options=CleanupOptions(max_removed_fraction=0.35))
    assert len(out) == 6


def test_energy_filter_drops_quiet_tail_blip():
    sr = 16000
    # Three loud notes plus one very quiet one at the same pitch.
    def tone(f, dur, amp):
        t = np.arange(int(sr * dur)) / sr
        env = np.minimum(1, t * 100) * np.exp(-t * 2)
        return (amp * np.sin(2 * np.pi * f * t) * env).astype(np.float32)

    x = np.concatenate([
        tone(440, 0.4, 0.8), np.zeros(int(sr * 0.1), dtype=np.float32),
        tone(440, 0.4, 0.75), np.zeros(int(sr * 0.1), dtype=np.float32),
        tone(440, 0.4, 0.7), np.zeros(int(sr * 0.1), dtype=np.float32),
        tone(440, 0.4, 0.01),
    ])
    notes = [
        note(0.0, 69), note(0.5, 69), note(1.0, 69), note(1.5, 69),
    ]
    out = cleanup(
        notes, wav=x, sample_rate=sr,
        options=CleanupOptions(drop_restrike_overlaps=False, merge_gap_s=0.0, min_duration_s=0.0, min_relative_db=-18.0),
    )
    assert len(out) == 3


def test_empty_input():
    assert cleanup([]) == []
