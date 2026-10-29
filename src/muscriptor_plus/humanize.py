"""Give a transcription a live feel: phrase dynamics and micro-timing.

Even with per-note velocities restored (see :mod:`muscriptor_plus.velocity`),
a transcription can still read as mechanical: every note sits exactly on the
grid, and a phrase that was played as one long crescendo comes back as a run of
unrelated velocities.

This module adds the two things that make a performance sound human without
inventing anything the recording doesn't support:

* phrase dynamics - notes are grouped into phrases at the silences between
  them, and each phrase is given a smooth dynamic arc (an arch, a rising
  crescendo or a falling diminuendo) shaped by the measured velocities. This is
  what turns "twelve notes at similar loudness" into "one gesture".
* micro-timing - a small, deterministic timing offset per note, seeded from the
  note's own position so the result is reproducible, and scaled by how strongly
  the note is played. A gentle rubato, not random jitter.

Both are conservative by default: they move nothing far, and every effect can
be dialled to zero.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Sequence

# Note objects are treated structurally: anything with onset/offset/pitch (and
# optionally velocity/program/is_drum) works, matching the rest of the package.


@dataclasses.dataclass
class HumanizeOptions:
    """How much live feel to add, and of which kind."""

    # --- phrase dynamics ---
    # Group notes into phrases at gaps of at least this many seconds.
    phrase_gap_s: float = 0.45
    # Arc applied to each phrase's velocities: "arch" (swell in the middle),
    # "crescendo" (grow), "diminuendo" (fade), or "none".
    phrase_shape: str = "arch"
    # Peak size of the arc, in velocity units, at the loudest point of the
    # phrase. 0 disables phrase shaping.
    phrase_depth: float = 10.0
    # Keep a phrase's own measured loudness centre; the arc is added on top
    # rather than replacing the velocities, so relative accents survive.
    # --- accents ---
    # Extra velocity for notes landing on a beat (and again on a downbeat).
    accent_beat: float = 4.0
    accent_downbeat: float = 6.0
    beats_per_bar: int | None = None
    bpm: float | None = None
    # --- micro-timing ---
    # Largest timing offset, in seconds, applied to a note. 0 disables it.
    timing_jitter_s: float = 0.012
    # Random seed for the deterministic jitter (same seed -> same output).
    seed: int = 0
    # --- safety ---
    min_velocity: int = 1
    max_velocity: int = 127

    def __post_init__(self):
        if self.phrase_shape not in ("arch", "crescendo", "diminuendo", "none"):
            raise ValueError(f"unknown phrase_shape: {self.phrase_shape!r}")
        if self.phrase_gap_s < 0 or self.phrase_depth < 0 or self.timing_jitter_s < 0:
            raise ValueError("gaps, depth and jitter must be non-negative")


def _deterministic_unit(*parts) -> float:
    """A stable pseudo-random value in [-1, 1] from the given parts.

    Hashing the note's own identity means the same note always gets the same
    offset, so re-running the transcription does not shuffle the timing.
    """
    key = "|".join(str(p) for p in parts).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(1 << 64)  # [0, 1)
    return value * 2.0 - 1.0


def _group_phrases(notes: Sequence, gap_s: float) -> list[list[int]]:
    """Indices of notes grouped into phrases, split at silences >= ``gap_s``.

    The gap is measured between one note's offset and the next note's onset
    (across all pitches), so a sustained chord does not split a phrase.
    """
    order = sorted(range(len(notes)), key=lambda i: notes[i].onset)
    phrases: list[list[int]] = []
    current: list[int] = []
    last_offset = None
    for i in order:
        note = notes[i]
        onset = float(note.onset)
        offset = float(getattr(note, "offset", onset) or onset)
        if current and last_offset is not None and onset - last_offset >= gap_s:
            phrases.append(current)
            current = []
        current.append(i)
        last_offset = offset if last_offset is None else max(last_offset, offset)
    if current:
        phrases.append(current)
    return phrases


def _phrase_arc(n: int, shape: str) -> list[float]:
    """Unit arc values ([-1, 1]) across a phrase of ``n`` notes."""
    if n <= 1 or shape == "none":
        return [0.0] * n
    xs = [i / (n - 1) for i in range(n)]
    if shape == "arch":
        return [math.sin(math.pi * x) for x in xs]
    if shape == "crescendo":
        return [x for x in xs]
    # diminuendo
    return [1.0 - x for x in xs]


class Humanizer:
    """Apply phrase dynamics, accents and micro-timing to a note list."""

    def __init__(self, options: HumanizeOptions | None = None):
        self.options = options or HumanizeOptions()

    def humanize(self, notes: Sequence) -> list:
        notes = list(notes)
        if not notes:
            return []
        velocities = self._velocities(notes)
        times = self._times(notes)
        out = []
        for i, note in enumerate(notes):
            out.append(self._rewrite(note, velocities[i], times[i]))
        return out

    # ------------------------------------------------------------------
    def _base_velocities(self, notes: Sequence) -> list[float]:
        return [float(getattr(n, "velocity", 100)) for n in notes]

    def _velocities(self, notes: Sequence) -> list[float]:
        opt = self.options
        vel = self._base_velocities(notes)

        if opt.phrase_depth > 0 and opt.phrase_shape != "none":
            for phrase in _group_phrases(notes, opt.phrase_gap_s):
                arc = _phrase_arc(len(phrase), opt.phrase_shape)
                # Centre the arc so shaping never shifts the phrase's overall
                # loudness, only its internal shape.
                mean_arc = sum(arc) / len(arc)
                for idx, a in zip(phrase, arc):
                    vel[idx] += opt.phrase_depth * (a - mean_arc)

        if opt.accent_beat or opt.accent_downbeat:
            vel = self._apply_accents(notes, vel)

        return [self._clamp(v) for v in vel]

    def _apply_accents(self, notes: Sequence, vel: list[float]) -> list[float]:
        opt = self.options
        bpm = opt.bpm
        if bpm is None:
            # Estimate a tempo from median onset spacing so accents still land
            # somewhere sensible when no grid was supplied.
            bpm = self._estimate_bpm(notes)
        if not bpm or bpm <= 0:
            return vel
        beat = 60.0 / bpm
        bar = (opt.beats_per_bar or 4)
        out = list(vel)
        for i, note in enumerate(notes):
            pos = float(note.onset) / beat
            nearest = round(pos)
            if abs(pos - nearest) > 0.15:
                continue  # not on a beat
            if bar > 1 and nearest % bar == 0:
                out[i] += opt.accent_downbeat
            else:
                out[i] += opt.accent_beat
        return out

    @staticmethod
    def _estimate_bpm(notes: Sequence) -> float | None:
        onsets = sorted(float(n.onset) for n in notes)
        if len(onsets) < 4:
            return None
        deltas = [b - a for a, b in zip(onsets, onsets[1:]) if b - a > 1e-3]
        if not deltas:
            return None
        deltas.sort()
        median = deltas[len(deltas) // 2]
        if median <= 0:
            return None
        bpm = 60.0 / median
        # Fold implausible estimates into a musical range.
        while bpm < 50:
            bpm *= 2
        while bpm > 220:
            bpm /= 2
        return bpm

    def _times(self, notes: Sequence) -> list[float]:
        opt = self.options
        if opt.timing_jitter_s <= 0:
            return [0.0] * len(notes)
        out = []
        for note in notes:
            unit = _deterministic_unit(
                opt.seed, round(float(note.onset), 3), int(note.pitch), int(getattr(note, "program", 0))
            )
            # Louder notes lead slightly, quieter ones lag: a subtle, musical
            # bias rather than symmetric noise.
            vel = float(getattr(note, "velocity", 100))
            bias = (vel - 64.0) / 64.0  # ~[-1, 1]
            out.append(opt.timing_jitter_s * (0.7 * unit - 0.3 * bias))
        return out

    # ------------------------------------------------------------------
    def _clamp(self, v: float) -> float:
        return float(min(self.options.max_velocity, max(self.options.min_velocity, round(v))))

    def _rewrite(self, note, velocity: float, dt: float):
        velocity = int(self._clamp(velocity))
        onset = float(note.onset) + dt
        offset = float(getattr(note, "offset", note.onset))
        if getattr(note, "is_drum", False):
            offset = onset
        else:
            offset = max(offset + dt, onset + 1e-3)
        if dataclasses.is_dataclass(note):
            try:
                return dataclasses.replace(note, onset=onset, offset=offset, velocity=velocity)
            except TypeError:
                pass
        import copy

        clone = copy.copy(note)
        clone.onset = onset
        clone.offset = offset
        clone.velocity = velocity
        return clone
