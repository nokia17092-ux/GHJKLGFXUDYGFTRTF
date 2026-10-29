"""Per-note dynamics (forte / piano) estimated from the recording.

MuScriptor's tokenizer has a *binary* velocity token: a note is either
"sounding" or "not sounding". Nothing in the model encodes how hard it was
played, and ``muscriptor.utils.midi.notes_to_midi`` therefore writes every note
at a constant ``velocity=100``. The transcription is rhythmically and
pitch-accurately right but dynamically flat.

This module recovers the performance's dynamics. For every transcribed note it
measures the energy of that note's own harmonics in the audio and maps it onto
the MIDI velocity range, so a chord hammered out at fortissimo comes back loud
and the same chord played pianissimo comes back soft, with the shape in between
preserved.

The estimate is deliberately absolute rather than per-piece-relative: the
mapping is anchored to a calibrated reference level (``center_db``), so a quiet
recording really does produce quieter velocities than a loud one. A robust
per-piece spread correction keeps the full velocity range in use even when the
recording is compressed or quiet overall.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
import torch

# MIDI note -> frequency reference.
_A4_HZ = 440.0
_A4_PITCH = 69

# Number of harmonics summed per note. Enough to capture the body of most
# instruments without reaching so high that neighbouring notes' partials
# dominate the measurement.
_MAX_HARMONICS = 8

# Energy below this (in linear magnitude) is silence; used to avoid log(0).
_EPS = 1e-10


@dataclasses.dataclass
class VelocityOptions:
    """How per-note velocities are estimated and mapped onto MIDI.

    The first group controls the measurement; the second maps the measured
    loudness onto the 1..127 MIDI velocity range.
    """

    # --- measurement ---
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 160  # 100 frames/second at 16 kHz
    # Energy is measured over this many seconds from the note onset (or the
    # whole note when it is shorter). Attack transients carry the dynamic, so a
    # short window is usually enough and keeps overlapping notes apart.
    onset_window_s: float = 0.06
    # Weight applied to the fundamental relative to its harmonics: 1.0 treats
    # every partial alike, <1.0 leans on the fundamental for a steadier pitch
    # estimate on quiet or breathy tones.
    fundamental_weight: float = 1.5
    # --- mapping ---
    min_velocity: int = 24
    max_velocity: int = 124
    # The loudness (in dB, with a full-scale sine at ~0 dB) that maps to the
    # middle of the velocity range, and how many dB span the whole range.
    # Larger ``dynamic_range_db`` = a gentler slope, i.e. less contrast.
    # -8 dB is roughly the RMS of a comfortably-loud single note.
    center_db: float = -8.0
    dynamic_range_db: float = 34.0
    # Auto-calibrate ``center_db`` from the piece's own loudness percentiles
    # instead of using the fixed reference above. Keeps the full velocity range
    # in play for recordings that sit far from the reference level.
    auto_calibrate: bool = True
    # Percentiles (low, high) of the measured loudness mapped onto
    # (min_velocity, max_velocity) when ``auto_calibrate`` is on. Using
    # percentiles rather than min/max stops a single stray hit from flattening
    # everything else.
    calibration_percentiles: tuple[float, float] = (5.0, 95.0)
    # Minimum contrast: after auto-calibration, if the notes span less than this
    # many dB the calibration is pulled back towards the fixed reference, so a
    # genuinely even performance isn't stretched into artificial contrast.
    min_span_db: float = 6.0
    # Drums have no pitch, so their "harmonic" energy is meaningless. Measure
    # them broadband instead (over the whole spectrum in the onset window).
    broadband_drums: bool = True
    # Clamp the final velocities into [1, 127] (always on) and round.
    clamp: bool = True

    def __post_init__(self):
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("n_fft and hop_length must be positive")
        if not 1 <= self.min_velocity <= self.max_velocity <= 127:
            raise ValueError("require 1 <= min_velocity <= max_velocity <= 127")
        if self.dynamic_range_db <= 0:
            raise ValueError("dynamic_range_db must be positive")
        lo, hi = self.calibration_percentiles
        if not 0.0 <= lo < hi <= 100.0:
            raise ValueError("calibration_percentiles must be (lo, hi) with 0<=lo<hi<=100")


def pitch_to_hz(pitch: int) -> float:
    """Equal-tempered MIDI pitch to frequency in Hz (A4 = 440 Hz)."""
    return _A4_HZ * (2.0 ** ((pitch - _A4_PITCH) / 12.0))


def _harmonic_bins(pitch: int, n_fft: int, sample_rate: int) -> list[tuple[int, float]]:
    """FFT bins carrying ``pitch``'s partials, as (bin index, weight) pairs.

    A partial is dropped once it passes Nyquist. The nearest bin to each
    harmonic is used, which is exact enough at this FFT size (bin width
    ``sample_rate / n_fft`` ~ 7.8 Hz at 16 kHz / 2048).
    """
    nyquist = sample_rate / 2.0
    bin_hz = sample_rate / n_fft
    f0 = pitch_to_hz(pitch)
    bins: dict[int, float] = {}
    for h in range(1, _MAX_HARMONICS + 1):
        freq = f0 * h
        if freq >= nyquist:
            break
        b = int(round(freq / bin_hz))
        if b <= 0:
            continue
        weight = 1.0 if h > 1 else 1.0  # per-harmonic weight applied by caller
        bins[b] = max(bins.get(b, 0.0), weight)
    return sorted(bins.items())


class VelocityEstimator:
    """Estimate a MIDI velocity for each transcribed note from the audio.

    Example::

        est = VelocityEstimator(sample_rate=16000)
        velocities = est.estimate(waveform, notes)   # one int per note
        notes = est.apply(waveform, notes)           # notes with .velocity set
    """

    def __init__(self, options: VelocityOptions | None = None, sample_rate: int | None = None):
        self.options = options or VelocityOptions()
        if sample_rate is not None:
            self.options = dataclasses.replace(self.options, sample_rate=int(sample_rate))
        self._mag: np.ndarray | None = None
        self._mag_wav_id: int | None = None

    # ------------------------------------------------------------------
    # Spectrogram
    # ------------------------------------------------------------------
    def _magnitude(self, wav: np.ndarray) -> np.ndarray:
        """Amplitude-normalized magnitude STFT, cached per waveform.

        Normalized so that a full-scale sine at frequency ``f`` reads an
        amplitude of ~1.0 at the bin nearest ``f``: the raw STFT magnitude of
        such a sine is ``A * window.sum() / 2``, so multiplying by
        ``2 / window.sum()`` puts the measurement in the same units as the
        waveform's own amplitude. Without this the dB values are only
        meaningful up to an arbitrary FFT-size-dependent constant, which would
        make the absolute velocity mapping (``center_db``) meaningless.
        """
        wav_id = id(wav)
        if self._mag is not None and self._mag_wav_id == wav_id:
            return self._mag
        wav_t = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
        if wav_t.dim() > 1:
            wav_t = wav_t.mean(dim=0)
        window = torch.hann_window(self.options.n_fft)
        spec = torch.stft(
            wav_t,
            n_fft=self.options.n_fft,
            hop_length=self.options.hop_length,
            win_length=self.options.n_fft,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
            normalized=False,
            onesided=True,
        )
        mag = spec.abs().numpy().astype(np.float32)  # [freq, frames]
        mag *= 2.0 / float(window.sum().item())
        self._mag = mag
        self._mag_wav_id = wav_id
        return mag

    def _frame_range(self, onset: float, offset: float | None) -> tuple[int, int]:
        """Frame indices covering the measurement window for one note.

        The window starts half an FFT frame *after* the onset. A centered STFT
        frame reported at time ``t`` actually mixes audio from
        ``t - n_fft/2`` to ``t + n_fft/2``, so the frame sitting exactly on the
        onset still contains the tail of the previous note. Skipping to the
        first frame whose window opens at the onset keeps the measurement to
        this note's own sound; without it a loud note following a quiet one
        leaks energy across the boundary and the two read as equally loud.
        """
        fps = self.options.sample_rate / self.options.hop_length
        pad_s = self.options.n_fft / (2.0 * self.options.sample_rate)
        start = int(math.floor((onset + pad_s) * fps))
        end = start + max(1, int(round(self.options.onset_window_s * fps)))
        if offset is not None and offset > onset:
            end = min(end, int(math.ceil((offset + pad_s) * fps)))
        start = max(0, start)
        return start, max(start + 1, end)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    def _note_loudness_db(self, mag: np.ndarray, wav: np.ndarray, note) -> float:
        """Loudness of one note in dB, from its own harmonics (or broadband)."""
        start, end = self._frame_range(note.onset, getattr(note, "offset", None))
        end = min(end, mag.shape[1])
        if start >= mag.shape[1]:
            return -np.inf
        block = mag[:, start:end]
        if block.size == 0:
            return -np.inf

        is_drum = getattr(note, "is_drum", False)
        if is_drum and self.options.broadband_drums:
            # A drum hit has no pitch, so its harmonics carry no meaning: its
            # loudness is the broadband level of the onset window. Measured as
            # time-domain RMS, which is on the same scale as the harmonic path
            # (both reduce to the signal's own amplitude for a tone), so the
            # same velocity mapping applies to both.
            pad = int(round(self.options.n_fft / 2))
            lo = start * self.options.hop_length - pad
            hi = end * self.options.hop_length - pad
            lo = max(0, lo)
            segment = wav[lo:hi]
            if segment.size == 0:
                return -np.inf
            rms = float(np.sqrt(np.mean(np.square(segment, dtype=np.float64))))
            return 20.0 * math.log10(max(rms, _EPS))
        else:
            bins = _harmonic_bins(int(note.pitch), self.options.n_fft, self.options.sample_rate)
            if not bins:
                energy = float(block.sum())
            else:
                idx = np.fromiter((b for b, _ in bins), dtype=np.int64, count=len(bins))
                idx = idx[idx < block.shape[0]]
                if idx.size == 0:
                    return -np.inf
                weights = np.ones(idx.size, dtype=np.float64)
                weights[0] = self.options.fundamental_weight
                sub = block[idx, :]
                # Per-frame sum of weighted harmonics, then peak over the
                # window: the attack peak is the note's dynamic, and taking the
                # peak (not the mean) keeps a fast decay from reading as quiet.
                per_frame = (sub * weights[:, None]).sum(axis=0)
                energy = float(per_frame.max())
        return 20.0 * math.log10(max(energy, _EPS))

    def _all_loudness(self, mag: np.ndarray, wav: np.ndarray, notes: Sequence) -> np.ndarray:
        return np.array(
            [self._note_loudness_db(mag, wav, n) for n in notes], dtype=np.float64
        )

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------
    def _calibrate(self, loud_db: np.ndarray) -> tuple[float, float]:
        """Return (center_db, half_range_db) mapping loudness onto velocity.

        Auto-calibration centres the velocity range on the piece's median
        loudness and spans its percentile spread, then falls back towards the
        fixed reference when the piece is too even to justify that spread.
        """
        opt = self.options
        finite = loud_db[np.isfinite(loud_db)]
        if not opt.auto_calibrate or finite.size == 0:
            return opt.center_db, opt.dynamic_range_db / 2.0

        lo_p, hi_p = opt.calibration_percentiles
        lo = float(np.percentile(finite, lo_p))
        hi = float(np.percentile(finite, hi_p))
        span = hi - lo
        if span < opt.min_span_db:
            # Too even to stretch: blend the observed span towards the nominal
            # one so the contrast stays modest instead of being amplified.
            span = opt.min_span_db
        half = max(span / 2.0, 1.0)

        # Centre on the observed loudness itself, so a quiet piece really does
        # come out quieter (it sits near the low end of the range), while the
        # span still follows the piece's own dynamics.
        center = float(np.median(finite))
        return center, half

    def _to_velocity(self, loud_db: np.ndarray, center: float, half: float) -> np.ndarray:
        opt = self.options
        # Normalized position in [-1, 1] (saturating) then mapped linearly onto
        # the velocity range. Linear in dB is a close match to perceived
        # loudness over this span.
        pos = np.clip((loud_db - center) / half, -1.0, 1.0)
        mid = (opt.min_velocity + opt.max_velocity) / 2.0
        half_v = (opt.max_velocity - opt.min_velocity) / 2.0
        vel = mid + pos * half_v
        vel = np.where(np.isfinite(loud_db), vel, mid)
        if opt.clamp:
            vel = np.clip(np.round(vel), 1, 127)
        return vel.astype(int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def estimate(self, wav: np.ndarray, notes: Sequence) -> np.ndarray:
        """Per-note MIDI velocities (int array, same length as ``notes``)."""
        notes = list(notes)
        if not notes:
            return np.zeros(0, dtype=int)
        wav = np.asarray(wav, dtype=np.float32)
        mag = self._magnitude(wav)
        loud = self._all_loudness(mag, wav, notes)
        center, half = self._calibrate(loud)
        return self._to_velocity(loud, center, half)

    def apply(self, wav: np.ndarray, notes: Sequence) -> list:
        """Return copies of ``notes`` with a ``velocity`` attribute set.

        Works on any object exposing ``onset``/``pitch`` (and optionally
        ``offset``/``is_drum``); the ``muscriptor.tokenizer.notes.Note``
        dataclass is the intended input. Notes without a ``velocity`` field get
        one added via :func:`dataclasses.replace` when possible, else the
        attribute is set in place on a shallow copy.
        """
        velocities = self.estimate(wav, notes)
        out = []
        for note, vel in zip(notes, velocities):
            if dataclasses.is_dataclass(note):
                try:
                    out.append(dataclasses.replace(note, velocity=int(vel)))
                    continue
                except TypeError:
                    pass
            clone = _shallow_clone(note)
            clone.velocity = int(vel)
            out.append(clone)
        return out


def _shallow_clone(obj):
    import copy

    return copy.copy(obj)
