"""Tests for the per-note-velocity MIDI writer."""

import io

from mido import MidiFile

from muscriptor_plus.midi_writer import VelNote, as_vel_notes, notes_to_midi


def vnote(onset, pitch, velocity, program=0, is_drum=False, offset=None, dur=0.4):
    return VelNote(
        is_drum=is_drum,
        program=program,
        onset=onset,
        offset=offset if offset is not None else onset + dur,
        pitch=pitch,
        velocity=velocity,
    )


def roundtrip(notes, **kwargs):
    midi = notes_to_midi(notes, **kwargs)
    buf = io.BytesIO()
    midi.save(file=buf)
    buf.seek(0)
    return MidiFile(file=buf)


def note_ons(midi):
    return [
        (msg.note, msg.velocity, msg.channel)
        for track in midi.tracks
        for msg in track
        if msg.type == "note_on"
    ]


def test_per_note_velocity_preserved():
    midi = roundtrip([vnote(0.0, 60, 40), vnote(0.5, 64, 120)])
    ons = dict((n, v) for n, v, _ in note_ons(midi))
    assert ons[60] == 40
    assert ons[64] == 120


def test_one_note_on_per_note():
    notes = [vnote(0.0, 60, 80), vnote(0.0, 64, 80), vnote(0.5, 67, 80)]
    midi = roundtrip(notes)
    assert len(note_ons(midi)) == 3


def test_note_offs_are_emitted():
    midi = roundtrip([vnote(0.0, 60, 80, offset=0.7)])
    offs = [m for tr in midi.tracks for m in tr if m.type == "note_off"]
    assert len(offs) == 1


def test_velocity_is_clamped():
    midi = roundtrip([vnote(0.0, 60, 300), vnote(0.5, 62, -5)])
    ons = {n: v for n, v, _ in note_ons(midi)}
    assert ons[60] == 127
    assert ons[62] == 1


def test_separate_tracks_per_program():
    midi = roundtrip([vnote(0.0, 60, 80, program=0), vnote(0.0, 40, 80, program=32)])
    channels = {c for _, _, c in note_ons(midi)}
    assert len(channels) == 2


def test_drums_on_channel_9():
    midi = roundtrip([vnote(0.0, 38, 100, is_drum=True)])
    assert all(c == 9 for _, _, c in note_ons(midi))


def test_repeated_pitch_released_before_restrike():
    # Same pitch twice in a row: the note-off must precede the second note-on.
    midi = roundtrip([vnote(0.0, 60, 80, offset=0.5), vnote(0.5, 60, 80, offset=1.0)])
    events = [
        (m.type, m.note)
        for tr in midi.tracks
        for m in tr
        if m.type in ("note_on", "note_off")
    ]
    assert events.count(("note_on", 60)) == 2
    assert events.count(("note_off", 60)) == 2
    # No two note_ons of the same pitch without a note_off in between.
    active = False
    for kind, pitch in events:
        if kind == "note_on":
            assert not active
            active = True
        else:
            assert active
            active = False


def test_as_vel_notes_default_and_passthrough():
    class Plain:
        def __init__(self, onset, pitch):
            self.onset = onset
            self.offset = onset + 0.3
            self.pitch = pitch
            self.program = 0
            self.is_drum = False

    out = as_vel_notes([Plain(0.0, 60)])
    assert out[0].velocity == 100
    keep = as_vel_notes([vnote(0.0, 60, 33)])
    assert keep[0].velocity == 33


def test_grid_options_do_not_break_writing():
    from muscriptor.utils.midi import ExportOptions

    midi = roundtrip(
        [vnote(0.0, 60, 80), vnote(0.5, 62, 70)],
        options=ExportOptions(bpm=120.0, beats_per_bar=4, quantize=True, subdivision=4),
    )
    assert len(note_ons(midi)) == 2
