"""Tests for multi-pass consensus merging."""

import pytest

from muscriptor_plus.consensus import (
    ConsensusDecoder,
    ConsensusOptions,
    _Candidate,
    _median,
    _mode_int,
)


def cand(pitch, onset, program=0, is_drum=False, offset=None, pass_index=0):
    return _Candidate(
        pitch=pitch,
        onset=onset,
        offset=offset if offset is not None else onset + 0.3,
        program=program,
        is_drum=is_drum,
        velocity=100,
        pass_index=pass_index,
    )


def test_median():
    assert _median([1.0, 2.0, 3.0]) == 2.0
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert _median([]) == 0.0


def test_mode_int():
    assert _mode_int([1, 2, 2, 3]) == 2
    assert _mode_int([5]) == 5
    assert _mode_int([]) == -1


def test_notes_agreed_by_majority_survive():
    # 3 passes all contain pitch 60; only 1 contains pitch 64.
    passes = [
        [cand(60, 0.0, pass_index=0)],
        [cand(60, 0.01, pass_index=1)],
        [cand(60, 0.02, pass_index=2), cand(64, 0.5, pass_index=2)],
    ]
    dec = ConsensusDecoder(ConsensusOptions(min_agreement=0.5))
    merged = dec.merge(passes)
    pitches = sorted(c.pitch for c in merged)
    assert pitches == [60]


def test_spurious_note_across_passes_is_removed():
    # A different spurious note in each pass must not accumulate votes.
    passes = [
        [cand(60, 0.0, pass_index=0), cand(61, 0.5, pass_index=0)],
        [cand(60, 0.0, pass_index=1), cand(62, 0.5, pass_index=1)],
        [cand(60, 0.0, pass_index=2), cand(63, 0.5, pass_index=2)],
    ]
    dec = ConsensusDecoder(ConsensusOptions(min_agreement=0.6))
    merged = dec.merge(passes)
    assert [c.pitch for c in merged] == [60]


def test_onset_is_median_of_agreeing_passes():
    passes = [
        [cand(60, 0.0, pass_index=0)],
        [cand(60, 0.02, pass_index=1)],
        [cand(60, 0.10, pass_index=2)],  # 0.10 is outside tolerance of 0.0
    ]
    dec = ConsensusDecoder(ConsensusOptions(min_agreement=0.5, onset_tolerance_s=0.05))
    merged = dec.merge(passes)
    assert len(merged) == 1
    assert merged[0].onset == pytest.approx(0.01)


def test_high_agreement_threshold_keeps_only_unanimous():
    passes = [
        [cand(60, 0.0, pass_index=0), cand(64, 0.4, pass_index=0)],
        [cand(60, 0.0, pass_index=1)],
    ]
    dec = ConsensusDecoder(ConsensusOptions(min_agreement=1.0))
    merged = dec.merge(passes)
    assert [c.pitch for c in merged] == [60]


def test_drums_and_pitched_notes_do_not_merge():
    passes = [
        [cand(38, 0.0, is_drum=True, pass_index=0)],
        [cand(38, 0.0, is_drum=False, pass_index=1)],
    ]
    dec = ConsensusDecoder(ConsensusOptions(min_agreement=0.5))
    # Different note kinds: each gets its own cluster, each with one vote,
    # which is below 0.5 * 2 = 1.0 only if strictly less. One vote each equals
    # the threshold, so both survive but stay separate.
    merged = dec.merge(passes)
    assert len(merged) == 2
    assert sum(c.is_drum for c in merged) == 1


def test_pitch_tolerance_absorbs_octave_adjacent_error():
    passes = [
        [cand(60, 0.0, pass_index=0)],
        [cand(61, 0.0, pass_index=1)],
    ]
    strict = ConsensusDecoder(ConsensusOptions(min_agreement=1.0, pitch_tolerance=0))
    assert strict.merge(passes) == []
    loose = ConsensusDecoder(ConsensusOptions(min_agreement=1.0, pitch_tolerance=1))
    assert len(loose.merge(passes)) == 1


def test_empty_passes():
    dec = ConsensusDecoder()
    assert dec.merge([]) == []
    assert dec.merge([[], []]) == []


def test_invalid_options_rejected():
    with pytest.raises(ValueError):
        ConsensusOptions(passes=())
    with pytest.raises(ValueError):
        ConsensusOptions(min_agreement=0.0)
    with pytest.raises(ValueError):
        ConsensusOptions(min_agreement=1.5)
