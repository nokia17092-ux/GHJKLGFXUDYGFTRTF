"""MIDI writer that carries per-note velocity.

``muscriptor.utils.midi.notes_to_midi`` writes one constant velocity for the
whole file, which throws away exactly the dynamics
:mod:`muscriptor_plus.velocity` worked to recover. Everything else that writer
does - resolving the beat grid, quantizing, filling gaps, the tempo map - is
worth keeping, so this module reuses those helpers and only replaces the final
``Note`` -> MIDI step with one that reads each note's own velocity.

``velocity`` is an extra field on :class:`VelNote`. Notes produced by MuScriptor
have no such field, so :func:`as_vel_notes` copies them in with a default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from mido import Message, MetaMessage, MidiFile, MidiTrack

from muscriptor.tokenizer.notes import DRUM_PROGRAM
from muscriptor.utils.beats import BAR_OFFSET_MARKER
from muscriptor.utils.midi import (
    TICKS_PER_BEAT,
    ExportOptions,
    gaps_filled,
    quantized_notes,
    resolve_grid,
    shifted_notes,
    timeline_for,
    leading_bars,
)

DEFAULT_VELOCITY = 100


def _track_key(note) -> int:
    """The MIDI track/program a note belongs to (drums collapse to DRUM_PROGRAM)."""
    return DRUM_PROGRAM if (note.is_drum or note.program == DRUM_PROGRAM) else note.program


@dataclass
class VelNote:
    """A :class:`muscriptor.tokenizer.notes.Note` plus a MIDI velocity."""

    is_drum: bool
    program: int
    onset: float
    offset: float
    pitch: int
    velocity: int = DEFAULT_VELOCITY


def as_vel_notes(notes, default_velocity: int = DEFAULT_VELOCITY) -> list[VelNote]:
    """Convert any Note-like objects to :class:`VelNote`.

    A note that already carries a velocity keeps it; otherwise
    ``default_velocity`` is used.
    """
    out = []
    for n in notes:
        out.append(
            VelNote(
                is_drum=bool(getattr(n, "is_drum", False)),
                program=int(getattr(n, "program", 0)),
                onset=float(n.onset),
                offset=float(getattr(n, "offset", n.onset)),
                pitch=int(n.pitch),
                velocity=int(getattr(n, "velocity", default_velocity)),
            )
        )
    return out


def notes_to_midi(
    notes: list[VelNote],
    program_names: dict[int, str] | None = None,
    beat_grid=None,
    options: ExportOptions | None = None,
    ticks_per_beat: int = TICKS_PER_BEAT,
    drum_hit_beats: float = 1 / 32,
):
    """Write ``notes`` (each with its own velocity) as a type-1 MIDI file.

    Mirrors ``muscriptor.utils.midi.notes_to_midi`` for grid handling; only the
    per-note velocity differs.
    """
    options = options or ExportOptions()
    notes = as_vel_notes(notes)
    grid = resolve_grid(beat_grid, options, [n.onset for n in notes])
    delay = grid.onset_delay
    notes = shifted_notes(notes, -delay)
    earliest = min([-delay] + [n.onset for n in notes])
    timeline = timeline_for(grid, earliest)
    # From here on, note times are in beats from tick 0.
    notes = [
        dataclasses.replace(
            n, onset=timeline.beats_at(n.onset), offset=timeline.beats_at(n.offset)
        )
        for n in notes
    ]

    if options.quantize:
        if grid.beat_subdivision is None:
            import logging

            logging.getLogger(__name__).warning(
                "not quantizing: no beat subdivision to snap the notes to"
            )
        else:
            notes = quantized_notes(notes, 1 / grid.beat_subdivision, options.strength)

    if options.fill_gaps > 0:
        notes = gaps_filled(notes, options.fill_gaps)

    if options.trim_leading_bars and notes:
        empty = leading_bars(notes, grid)
        timeline = timeline.shifted(-empty)
        notes = shifted_notes(notes, -empty)

    return _vel_notes_to_midi(
        notes,
        ticks_per_beat=ticks_per_beat,
        program_names=program_names,
        beats_per_bar=grid.beats_per_bar,
        offset_s=-timeline.seconds_at(0.0),
        to_tick=lambda beats: round(beats * ticks_per_beat),
        tempo_changes=timeline.tempo_changes(ticks_per_beat),
        drum_length=drum_hit_beats,
    )


def _vel_notes_to_midi(
    notes: list[VelNote],
    ticks_per_beat: int = TICKS_PER_BEAT,
    program_names: dict[int, str] | None = None,
    beats_per_bar: int | None = None,
    offset_s: float = 0.0,
    to_tick=None,
    tempo_changes: list[tuple[int, int]] | None = None,
    drum_length: float = 0.01,
) -> MidiFile:
    """Type-1 MIDI from notes that carry their own velocity.

    Layout matches ``muscriptor.tokenizer.notes.note_event2midi``: one track per
    program, drums on channel 9, other programs on channels 0-8 then 10-15.
    """
    from muscriptor.tokenizer.notes import _append_tempo_changes

    if to_tick is None:

        def to_tick(time: float) -> int:
            from mido import second2tick

            return round(second2tick(time + offset_s, ticks_per_beat, 500000))

    tempo_changes = tempo_changes or [(0, 500000)]

    midi = MidiFile(ticks_per_beat=ticks_per_beat, type=1)
    meta_track = MidiTrack()
    if beats_per_bar is not None:
        meta_track.append(
            MetaMessage("time_signature", numerator=beats_per_bar, denominator=4, time=0)
        )
    if offset_s:
        meta_track.append(
            MetaMessage("marker", text=f"{BAR_OFFSET_MARKER}{offset_s:.4f}", time=0)
        )
    _append_tempo_changes(meta_track, tempo_changes, [])
    midi.tracks.append(meta_track)

    # Assign a channel to each program up front, in order of first appearance.
    program_to_channel: dict[int, int] = {}
    available_channels = list(range(0, 9)) + list(range(10, 16))
    for note in sorted(notes, key=lambda n: n.onset):
        key = _track_key(note)
        if key not in program_to_channel:
            if key == DRUM_PROGRAM:
                program_to_channel[key] = 9
            else:
                try:
                    program_to_channel[key] = available_channels.pop(0)
                except IndexError:
                    program_to_channel[key] = 15

    # One note_on at the onset and one note_off at the end, per note.
    tracks: dict[int, list[tuple[int, Message]]] = {}
    for note in notes:
        key = _track_key(note)
        channel = program_to_channel[key]
        end_beats = note.onset + drum_length if note.is_drum else note.offset
        onset_tick = to_tick(note.onset)
        end_tick = max(onset_tick + 1, to_tick(end_beats))
        bucket = tracks.setdefault(key, [])
        bucket.append(
            (
                onset_tick,
                Message(
                    "note_on",
                    note=int(note.pitch),
                    velocity=int(max(1, min(127, note.velocity))),
                    channel=channel,
                ),
            )
        )
        bucket.append(
            (
                end_tick,
                Message("note_off", note=int(note.pitch), velocity=0, channel=channel),
            )
        )

    for key in tracks:
        messages = tracks[key]
        # Note-offs before note-ons at the same tick so a repeated pitch is
        # released before it is struck again.
        messages.sort(key=lambda t: (t[0], 0 if t[1].type == "note_off" else 1))
        track = MidiTrack()
        midi.tracks.append(track)
        channel = program_to_channel[key]
        if key == DRUM_PROGRAM:
            name = (program_names or {}).get(key, "drums")
            gm_program = 0
        else:
            name = (program_names or {}).get(key, f"program {key}")
            gm_program = key
        track.append(MetaMessage("track_name", name=name, time=0))
        track.append(Message("program_change", program=gm_program, time=0, channel=channel))
        _append_tempo_changes(track, tempo_changes, messages)

    return midi
