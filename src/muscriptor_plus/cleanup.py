"""Remove likely transcription artifacts from dense or unclear passages.

MuScriptor's decoder is permissive by design: when it is unsure it would rather
emit a note than drop one. In a dense or noisy passage that shows up as notes
that are almost certainly not there - a one-frame blip on a note's decay tail,
a duplicate of a pitch already sounding, a burst of a dozen notes at the exact
same instant.

This module removes those, using only information the transcription already
carries plus the audio. Every rule is conservative and independently
switchable, because a wrong deletion is worse than a wrong insertion for
someone reading the result as a score.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class CleanupOptions:
    """Which artifact filters to apply and how aggressive they are."""

    # Drop a note shorter than this, unless it is a drum hit. A real note is
    # almost never this short; a decoder blip is.
    min_duration_s: float = 0.045
    # Drop a note whose onset falls inside another note of the same pitch on
    # the same instrument (the model re-striking a note that is still sounding).
    drop_restrike_overlaps: bool = True
    # Collapse a run of notes of the same pitch on the same instrument that are
    # close enough together that they are one sustained note with blips.
    merge_gap_s: float = 0.04
    # Drop a note whose measured energy is this many dB below the median note
    # of the same instrument: almost certainly a tail or a mis-trigger. Needs
    # the audio; skipped when no waveform is supplied.
    min_relative_db: float | None = -18.0
    # Never delete more than this fraction of the notes, whatever the rules
    # say: a guard against a filter that misfires on an unusual piece. When it
    # trips, the original notes are returned unchanged.
    max_removed_fraction: float = 0.35
    # The guard is meaningless on a handful of notes (removing one of two is
    # 50%), so it only applies once there are at least this many.
    min_notes_for_guard: int = 6


def _as_note_like(note, velocity):
    return note


def cleanup(
    notes,
    wav=None,
    sample_rate: int = 16000,
    options: CleanupOptions | None = None,
) -> list:
    """Return ``notes`` with likely artifacts removed.

    ``notes`` may be ``VelNote`` objects (with ``velocity``) or plain
    ``Note`` objects; whatever comes in is what comes out. ``wav`` is optional
    and only enables the energy-based filter.
    """
    options = options or CleanupOptions()
    notes = list(notes)
    if not notes:
        return []
    original_count = len(notes)

    kept = notes
    if options.drop_restrike_overlaps:
        kept = _drop_restrikes(kept)
    if options.merge_gap_s > 0:
        kept = _merge_blips(kept, options.merge_gap_s)
    kept = _drop_short(kept, options.min_duration_s)
    if wav is not None and options.min_relative_db is not None:
        kept = _drop_quiet_outliers(kept, wav, sample_rate, options.min_relative_db)

    # Guard: if the rules were too aggressive, return the input unchanged
    # rather than silently mangling a short or unusual transcription.
    if (
        original_count >= options.min_notes_for_guard
        and (original_count - len(kept)) / original_count > options.max_removed_fraction
    ):
        return notes
    return kept


def _drop_short(notes, min_duration_s: float) -> list:
    out = []
    for note in notes:
        if getattr(note, "is_drum", False):
            out.append(note)
            continue
        duration = float(getattr(note, "offset", note.onset)) - float(note.onset)
        if duration >= min_duration_s:
            out.append(note)
    return out


def _drop_restrikes(notes) -> list:
    """Drop a note that starts while the same pitch is still sounding."""
    groups: dict[tuple[int, int, bool], list] = defaultdict(list)
    for note in notes:
        groups[(int(note.pitch), int(getattr(note, "program", 0)), bool(getattr(note, "is_drum", False)))].append(note)

    kept = []
    for group in groups.values():
        group.sort(key=lambda n: n.onset)
        last_offset = None
        for note in group:
            onset = float(note.onset)
            if last_offset is not None and onset < last_offset - 1e-6:
                continue  # starts inside the previous note: a restrike artifact
            kept.append(note)
            last_offset = max(last_offset or 0.0, float(getattr(note, "offset", onset)))
    kept.sort(key=lambda n: (n.onset, n.pitch))
    return kept


def _merge_blips(notes, gap_s: float) -> list:
    """Fuse same-pitch notes separated by less than ``gap_s`` into one."""
    groups: dict[tuple[int, int, bool], list] = defaultdict(list)
    for note in notes:
        groups[(int(note.pitch), int(getattr(note, "program", 0)), bool(getattr(note, "is_drum", False)))].append(note)

    kept = []
    for group in groups.values():
        group.sort(key=lambda n: n.onset)
        current = group[0]
        for nxt in group[1:]:
            gap = float(nxt.onset) - float(getattr(current, "offset", current.onset))
            if gap <= gap_s:
                current = dataclasses.replace(
                    current,
                    offset=max(float(getattr(current, "offset", current.onset)), float(getattr(nxt, "offset", nxt.onset))),
                )
            else:
                kept.append(current)
                current = nxt
        kept.append(current)
    kept.sort(key=lambda n: (n.onset, n.pitch))
    return kept


def _drop_quiet_outliers(notes, wav, sample_rate: int, min_relative_db: float) -> list:
    """Drop non-drum notes far quieter than the median note of their instrument."""
    from muscriptor_plus.velocity import VelocityEstimator, VelocityOptions

    estimator = VelocityEstimator(VelocityOptions(auto_calibrate=False, sample_rate=sample_rate))
    mag = estimator._magnitude(np.asarray(wav, dtype=np.float32))
    loud = estimator._all_loudness(mag, np.asarray(wav, dtype=np.float32), notes)

    by_program: dict[tuple[int, bool], list[int]] = defaultdict(list)
    for i, note in enumerate(notes):
        by_program[(int(getattr(note, "program", 0)), bool(getattr(note, "is_drum", False)))].append(i)

    drop = set()
    for key, indices in by_program.items():
        if key[1]:  # never energy-filter drums
            continue
        finite = [loud[i] for i in indices if np.isfinite(loud[i])]
        if len(finite) < 3:
            continue
        reference = float(np.median(finite))
        for i in indices:
            if np.isfinite(loud[i]) and loud[i] - reference < min_relative_db:
                drop.add(i)
    return [note for i, note in enumerate(notes) if i not in drop]
