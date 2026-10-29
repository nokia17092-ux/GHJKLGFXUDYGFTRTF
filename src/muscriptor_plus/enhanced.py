"""One entry point that runs MuScriptor with every enhancement applied.

``EnhancedTranscriber`` wraps a loaded ``muscriptor.TranscriptionModel`` and
adds, in order:

1. transcription with the caller's decoding options (optionally via the
   multi-pass :class:`~muscriptor_plus.consensus.ConsensusDecoder`);
2. per-note dynamics from the audio (:mod:`muscriptor_plus.velocity`);
3. phrase shaping and micro-timing (:mod:`muscriptor_plus.humanize`);
4. MIDI written with those per-note velocities
   (:mod:`muscriptor_plus.midi_writer`).

Nothing here changes the model's weights or its note decisions: it is all
post-processing plus optional multi-pass decoding, so it is safe to compare
against the stock pipeline and to switch off piece by piece.
"""

from __future__ import annotations

import io
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.tokenizer.notes import DRUM_PROGRAM, Note, validate_notes
from muscriptor.tokenizer.notes import trim_overlapping_notes
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils.audio import load_audio
from muscriptor.utils.midi import ExportOptions

from muscriptor_plus.cleanup import CleanupOptions, cleanup
from muscriptor_plus.consensus import ConsensusDecoder, ConsensusOptions, _Candidate
from muscriptor_plus.humanize import HumanizeOptions, Humanizer
from muscriptor_plus.midi_writer import as_vel_notes, notes_to_midi
from muscriptor_plus.velocity import VelocityEstimator, VelocityOptions

_SAMPLE_RATE = 16000


@dataclass
class EnhancedResult:
    """The outcome of one enhanced transcription."""

    notes: list  # VelNote list, velocities already applied
    midi: bytes
    beat_grid: object | None = None
    # Number of decoding passes actually run (1 unless consensus is on).
    passes: int = 1
    # Per-note velocities as ints, parallel to ``notes``.
    velocities: list[int] = field(default_factory=list)

    def save_midi(self, path) -> None:
        with open(path, "wb") as fh:
            fh.write(self.midi)


class EnhancedTranscriber:
    """Run MuScriptor with dynamics, live feel and optional consensus decoding.

    Example::

        from muscriptor.transcription_model import TranscriptionModel
        from muscriptor_plus import EnhancedTranscriber

        model = TranscriptionModel.load_model("large")
        enh = EnhancedTranscriber(model)
        result = enh.transcribe("song.wav")
        result.save_midi("song.mid")
    """

    def __init__(
        self,
        model: TranscriptionModel,
        velocity: VelocityOptions | None = None,
        humanize: HumanizeOptions | None = None,
        consensus: ConsensusOptions | None = None,
        cleanup: CleanupOptions | None = None,
    ):
        self.model = model
        self.velocity_options = velocity or VelocityOptions()
        self.humanize_options = humanize or HumanizeOptions()
        self.consensus_options = consensus
        self.cleanup_options = cleanup or CleanupOptions()
        self._estimator = VelocityEstimator(self.velocity_options)
        self._humanizer = Humanizer(self.humanize_options)
        self._decoder = ConsensusDecoder(consensus) if consensus else None

    # ------------------------------------------------------------------
    def _load_wav(self, audio) -> np.ndarray:
        """Mono float32 numpy at the estimator's sample rate."""
        if isinstance(audio, tuple):
            tensor, sample_rate = audio
            wav = tensor
            if hasattr(wav, "detach"):
                wav = wav.detach().cpu().numpy()
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim == 2:
                wav = wav.mean(axis=0)
            if sample_rate and sample_rate != self.velocity_options.sample_rate:
                from muscriptor.utils.audio import resample
                import torch

                t = torch.from_numpy(wav)[None, :]
                wav = resample(t, sample_rate, self.velocity_options.sample_rate)[0].numpy()
        else:
            wav = load_audio(audio, target_sr=self.velocity_options.sample_rate)[0].numpy()
        return np.ascontiguousarray(wav, dtype=np.float32)

    # ------------------------------------------------------------------
    def _events_from_notes(self, notes: list[Note]) -> Iterator:
        """Turn decoded notes back into an event stream (for the stock writer)."""
        index = 0
        for note in notes:
            start = NoteStartEvent(
                pitch=note.pitch,
                start_time=note.onset,
                index=index,
                instrument="drums" if note.is_drum else self._instrument_name(note.program),
            )
            index += 1
            yield start
            yield NoteEndEvent(end_time=note.offset, start_event=start)

    def _instrument_name(self, program: int) -> str:
        try:
            return self.model._instrument_for_program(program)
        except Exception:
            return f"program_{program}"

    def _notes_from_events(self, events) -> list[Note]:
        """Collapse NoteStart/NoteEnd events into Notes (no velocity yet)."""
        open_notes: dict[int, Note] = {}
        notes: list[Note] = []
        for ev in events:
            if isinstance(ev, ProgressEvent):
                continue
            if isinstance(ev, NoteStartEvent):
                is_drum = ev.instrument == "drums"
                program = (
                    DRUM_PROGRAM if is_drum else self.model._program_for_instrument(ev.instrument)
                )
                open_notes[ev.index] = Note(
                    is_drum=is_drum,
                    program=program,
                    onset=ev.start_time,
                    offset=ev.start_time,
                    pitch=ev.pitch,
                )
            elif isinstance(ev, NoteEndEvent):
                note = open_notes.pop(ev.start_event.index, None)
                if note is not None:
                    note.offset = ev.end_time
                    notes.append(note)
        notes = validate_notes(notes, fix=True)
        notes = trim_overlapping_notes(notes, sort=True)
        return notes

    # ------------------------------------------------------------------
    def _decode_pass(self, audio, use_sampling, temperature, cfg_coef) -> list[Note]:
        events = list(
            self.model.transcribe(
                audio,
                use_sampling=use_sampling,
                temperature=temperature,
                cfg_coef=cfg_coef,
            )
        )
        return self._notes_from_events(events)

    def _transcribe_notes(self, audio, base_kwargs) -> tuple[list[Note], int]:
        """Decode notes, using consensus when configured."""
        if self._decoder is None:
            notes = self._decode_pass(
                audio,
                base_kwargs.get("use_sampling", False),
                base_kwargs.get("temperature", 1.0),
                base_kwargs.get("cfg_coef", 1.0),
            )
            return notes, 1

        passes: list[list[_Candidate]] = []
        for use_sampling, temperature, cfg_coef in self.consensus_options.passes:
            print(
                f"[muscriptor-plus] consensus pass (sampling={use_sampling}, "
                f"temp={temperature}, cfg={cfg_coef})",
                file=sys.stderr,
            )
            notes = self._decode_pass(audio, use_sampling, temperature, cfg_coef)
            passes.append(self._to_candidates(notes))
        merged = self._decoder.merge(passes)
        return self._candidates_to_notes(merged), len(passes)

    def _to_candidates(self, notes: list[Note]) -> list[_Candidate]:
        return [
            _Candidate(
                pitch=n.pitch,
                onset=n.onset,
                offset=n.offset,
                program=n.program,
                is_drum=n.is_drum,
                velocity=100,
                pass_index=0,
            )
            for n in notes
        ]

    def _candidates_to_notes(self, cands: list[_Candidate]) -> list[Note]:
        notes = [
            Note(
                is_drum=c.is_drum,
                program=c.program,
                onset=c.onset,
                offset=c.offset,
                pitch=c.pitch,
            )
            for c in cands
        ]
        return trim_overlapping_notes(validate_notes(notes, fix=True), sort=True)

    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio,
        use_sampling: bool = False,
        temperature: float = 1.0,
        cfg_coef: float = 1.0,
        export: ExportOptions | None = None,
        apply_velocity: bool = True,
        apply_humanize: bool = True,
        apply_cleanup: bool = True,
    ) -> EnhancedResult:
        """Transcribe ``audio`` with all enabled enhancements.

        Args:
            audio: a path, or a ``(waveform, sample_rate)`` tuple, as accepted
                by :meth:`muscriptor.TranscriptionModel.transcribe`.
            export: how to write the MIDI (grid corrections, quantizing, ...).
            apply_velocity / apply_humanize / apply_cleanup: switch the
                post-processing steps off individually.
        """
        base_kwargs = {
            "use_sampling": use_sampling,
            "temperature": temperature,
            "cfg_coef": cfg_coef,
        }
        notes, n_passes = self._transcribe_notes(audio, base_kwargs)
        wav = self._load_wav(audio)

        vel_notes = as_vel_notes(notes)
        if apply_cleanup and vel_notes:
            vel_notes = cleanup(
                vel_notes,
                wav=wav,
                sample_rate=self.velocity_options.sample_rate,
                options=self.cleanup_options,
            )
        if apply_velocity and vel_notes:
            vel_notes = self._estimator.apply(wav, vel_notes)
        if apply_humanize and vel_notes:
            vel_notes = self._humanizer.humanize(vel_notes)

        beat_grid = None
        try:
            beat_grid = self.model.detect_beat_grid_for(audio, "best-effort")
        except Exception as e:  # noqa: BLE001 - grid is best-effort
            print(f"[muscriptor-plus] beat detection skipped: {e}", file=sys.stderr)

        midi_file = notes_to_midi(
            vel_notes,
            program_names=self._program_names(vel_notes),
            beat_grid=beat_grid,
            options=export or ExportOptions(),
        )
        buf = io.BytesIO()
        midi_file.save(file=buf)

        return EnhancedResult(
            notes=vel_notes,
            midi=buf.getvalue(),
            beat_grid=beat_grid,
            passes=n_passes,
            velocities=[int(getattr(n, "velocity", 100)) for n in vel_notes],
        )

    def _program_names(self, notes) -> dict[int, str]:
        names: dict[int, str] = {}
        for n in notes:
            program = n.program
            if program not in names:
                names[program] = "drums" if n.is_drum else self._instrument_name(program).replace("_", " ")
        return names
