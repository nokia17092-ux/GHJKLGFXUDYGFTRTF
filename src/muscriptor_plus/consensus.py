"""Multi-pass consensus decoding for dense or unclear passages.

MuScriptor decodes greedily by default. On a clean, sparse passage that is
fine. On a dense mix - a wash of strings under a busy piano, a fast run, a
badly compressed recording - greedy decoding makes one unrecoverable choice per
token, and the errors show up as notes that are missing, doubled, or split
across the wrong instrument.

This module decodes the same audio several ways and merges the results:

* several decoding settings (greedy, a few sampling temperatures / guidance
  levels) produce candidate note sets;
* a note is kept when enough passes agree on it, where agreement means the same
  pitch within a small time window on a compatible instrument;
* the surviving notes take their timing from the passes that found them (the
  median onset, which cancels per-pass timing bias) and their velocity from the
  velocity estimator, not from the passes.

Averaging over independent decodes removes the single-choice errors while
keeping the notes every pass agrees on, which is exactly the confident part of
the transcription. Passes run with ``prelude_forcing`` (chunk-by-chunk) so
agreement also carries across chunk boundaries.

Because it needs N full decodes, it is the slowest option here; it is meant for
the passages that need it, not the whole piece (see ``regions``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

# A candidate note from one pass, reduced to what consensus needs.
@dataclass
class _Candidate:
    pitch: int
    onset: float
    offset: float
    program: int
    is_drum: bool
    velocity: int
    pass_index: int


@dataclasses.dataclass
class ConsensusOptions:
    """How many passes to run and how strict the merge is."""

    # Decoding settings for each pass. Each tuple is
    # (use_sampling, temperature, cfg_coef). Greedy first: it is the most
    # accurate single pass, so it anchors the vote.
    passes: tuple[tuple[bool, float, float], ...] = (
        (False, 1.0, 1.0),
        (False, 1.0, 1.3),
        (True, 0.7, 1.0),
        (True, 1.0, 1.2),
        (True, 1.2, 1.5),
    )
    # Two notes from different passes are "the same note" when their pitches
    # match and their onsets are within this many seconds.
    onset_tolerance_s: float = 0.05
    # An offset closer than this is treated as agreement too; notes are merged
    # to the median of the agreed offsets.
    offset_tolerance_s: float = 0.15
    # Minimum fraction of passes that must contain a note for it to survive.
    # 0.5 = a strict majority; raise it for fewer, safer notes.
    min_agreement: float = 0.5
    # Merge notes whose pitches are this many semitones apart (0 = exact only).
    # A small value (1) absorbs octave-adjacent pitch errors in dense chords.
    pitch_tolerance: int = 0
    # When merging, keep the instrument only if the agreeing passes agree on
    # it; otherwise fall back to the most common one.
    strict_instrument: bool = False
    # Trim leading/trailing silences before decoding, and ignore notes found
    # outside the requested regions.
    regions: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self):
        if not self.passes:
            raise ValueError("need at least one decoding pass")
        if not 0.0 < self.min_agreement <= 1.0:
            raise ValueError("min_agreement must be in (0, 1]")
        if self.onset_tolerance_s < 0 or self.pitch_tolerance < 0:
            raise ValueError("tolerances must be non-negative")


def _candidates_from_events(events, pass_index: int) -> list[_Candidate]:
    """Collapse a NoteStart/NoteEnd event stream into candidate notes."""
    from muscriptor.events import NoteStartEvent, NoteEndEvent

    from muscriptor.tokenizer.notes import DRUM_PROGRAM

    open_notes = {}
    out: list[_Candidate] = []
    for ev in events:
        if isinstance(ev, NoteStartEvent):
            open_notes[ev.index] = ev
        elif isinstance(ev, NoteEndEvent):
            start = ev.start_event
            is_drum = start.instrument == "drums"
            out.append(
                _Candidate(
                    pitch=start.pitch,
                    onset=float(start.start_time),
                    offset=float(ev.end_time),
                    program=DRUM_PROGRAM if is_drum else -1,
                    is_drum=is_drum,
                    velocity=100,
                    pass_index=pass_index,
                )
            )
    return out


def _program_of(start_event, model) -> int:
    if start_event.instrument == "drums":
        from muscriptor.tokenizer.notes import DRUM_PROGRAM

        return DRUM_PROGRAM
    try:
        return model._program_for_instrument(start_event.instrument)
    except Exception:
        return -1


class ConsensusDecoder:
    """Merge several decodes of the same audio into one confident note set.

    ``decode`` takes a callable that runs one pass (see
    :class:`muscriptor_plus.enhanced.EnhancedTranscriber`, which supplies it)
    and returns the merged notes.
    """

    def __init__(self, options: ConsensusOptions | None = None):
        self.options = options or ConsensusOptions()

    def merge(self, passes: list[list[_Candidate]]) -> list[_Candidate]:
        """Merge candidate lists (one per pass) into a consensus note list."""
        opt = self.options
        total = len(passes)
        if total == 0:
            return []
        # Flatten with a stable ordering so the merge is deterministic.
        flat = [c for p in passes for c in p]
        if not flat:
            return []
        flat.sort(key=lambda c: (c.onset, c.pitch, c.program, c.pass_index))

        threshold = opt.min_agreement * total
        clusters: list[list[_Candidate]] = []
        for cand in flat:
            target = None
            for cluster in reversed(clusters):
                if self._same_note(cluster[0], cand):
                    target = cluster
                    break
            if target is None:
                clusters.append([cand])
            else:
                target.append(cand)

        merged: list[_Candidate] = []
        for cluster in clusters:
            votes = len({c.pass_index for c in cluster})
            if votes < threshold:
                continue
            merged.append(self._reduce(cluster))
        merged.sort(key=lambda c: (c.onset, c.pitch))
        return merged

    def _same_note(self, a: _Candidate, b: _Candidate) -> bool:
        opt = self.options
        if a.is_drum != b.is_drum:
            return False
        if abs(a.pitch - b.pitch) > opt.pitch_tolerance:
            return False
        if abs(a.onset - b.onset) > opt.onset_tolerance_s:
            return False
        if not a.is_drum and a.program >= 0 and b.program >= 0 and a.program != b.program:
            # Different instruments may still be the same note if the passes
            # disagree on the instrument; allow it unless strict.
            if opt.strict_instrument:
                return False
        return True

    def _reduce(self, cluster: list[_Candidate]) -> _Candidate:
        onsets = sorted(c.onset for c in cluster)
        offsets = sorted(c.offset for c in cluster)
        pitches = [c.pitch for c in cluster]
        programs = [c.program for c in cluster if c.program >= 0]
        median_onset = _median(onsets)
        median_offset = _median(offsets)
        pitch = _mode_int(pitches)
        program = _mode_int(programs) if programs else -1
        is_drum = sum(c.is_drum for c in cluster) * 2 >= len(cluster)
        return _Candidate(
            pitch=pitch,
            onset=median_onset,
            offset=max(median_offset, median_onset + 0.01),
            program=program,
            is_drum=is_drum,
            velocity=int(round(sum(c.velocity for c in cluster) / len(cluster))),
            pass_index=cluster[0].pass_index,
        )


def _median(sorted_values: list[float]) -> float:
    n = len(sorted_values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def _mode_int(values: list[int]) -> int:
    if not values:
        return -1
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: (counts[k], -k))
