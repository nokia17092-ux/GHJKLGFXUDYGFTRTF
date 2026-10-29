# muscriptor-plus

Enhancements for [MuScriptor](https://huggingface.co/MuScriptor) music
transcription: real note dynamics, a live feel, and cleaner handling of dense
or unclear passages.

MuScriptor is a strong multi-instrument transcriber, but its tokenizer has a
*binary* velocity token - a note is either sounding or not. Nothing in the model
encodes how hard it was played, and its MIDI writer therefore emits every note
at the same velocity. The result is rhythmically and pitch-accurately right but
dynamically flat, and in a dense mix its greedy decoder leaves behind the usual
single-choice errors. This package fixes those three things without touching the
model's weights, so it layers cleanly on top of a stock MuScriptor install.

## What it does

| Module | Problem it solves |
|---|---|
| `velocity` | Restores **forte / piano**: measures each note's own harmonics in the recording and maps that loudness onto MIDI velocity. A hammered chord comes back loud, the same chord played pianissimo comes back soft. |
| `humanize` | Adds a **live feel**: groups notes into phrases and gives each a smooth dynamic arc (arch / crescendo / diminuendo), accents beats, and applies a small, deterministic micro-timing rubato. |
| `consensus` | Handles **dense or unclear passages**: decodes the audio several ways (greedy + a few sampling / guidance settings) and keeps only the notes a majority of passes agree on, so single-choice errors drop out. |
| `cleanup` | Removes likely **artifacts**: one-frame blips on a decay tail, restrikes of a note still sounding, same-pitch blips fused into one sustained note, and notes far quieter than their instrument's median. |

`EnhancedTranscriber` runs all of them behind one call.

## Install

```bash
pip install -e .
```

This pulls in `muscriptor` (and its `torch`). To use the `large` weights you
also need to accept the model's licence and download them; `tools/download_model.py`
will fetch them from the official gated repo when `HF_TOKEN` is set, or from the
public mirror that carries the identical weights otherwise.

## Use

```python
from muscriptor.transcription_model import TranscriptionModel
from muscriptor_plus import EnhancedTranscriber

model = TranscriptionModel.load_model("large")   # or a local path
enh = EnhancedTranscriber(model)
result = enh.transcribe("song.wav")
result.save_midi("song.mid")                     # per-note velocities included
```

Command line:

```bash
muscriptor-plus transcribe song.wav -o song.mid
muscriptor-plus transcribe dense.wav --consensus --consensus-passes 5
muscriptor-plus transcribe rubato.wav --phrase-shape crescendo --phrase-depth 14
muscriptor-plus transcribe clean.wav --no-cleanup --no-humanize
```

Every enhancement can be switched off individually, so the output is always
comparable against the stock pipeline.

## How the dynamics work

For each transcribed note the estimator takes a short magnitude STFT, sums the
energy in that note's own harmonic bins over a window starting half an FFT frame
after the onset, and converts it to dB on an absolute scale (a full-scale sine
reads ~0 dB). Two details matter:

* the half-frame offset - a centered STFT frame on the onset still contains the
  previous note's tail, which would make a quiet note after a loud one read as
  loud;
* the absolute scale - so a quiet recording really does produce quieter
  velocities than a loud one, rather than every piece being normalised to the
  same loudness.

A robust per-piece calibration then spreads the measured loudness over the MIDI
velocity range, with a guard that stops a genuinely even performance from being
stretched into artificial contrast.

Drums have no pitch, so their loudness is measured as broadband RMS over the
onset window, which is on the same scale as the harmonic measurement.

## Testing

```bash
pytest                      # fast unit tests, no model needed
pytest -m slow              # end-to-end against the real model (needs weights)
```

The unit tests synthesize their own audio rather than mocking: the estimator's
job is to measure real energy in real sound, so a fake spectrogram would test
nothing. The `slow` tests load the real model and check that a loud phrase and a
quiet phrase come out with distinct velocities.

## Scope and limits

This is post-processing plus multi-pass decoding, not a retrained model. It
cannot recover notes the model never emitted, and it does not change the model's
instrument classification. Genuine weight-level improvements - a learned
velocity head, fine-tuning on hard material - need a GPU, training data and the
training-time encoder, which is not part of the released inference package; see
[FINE_TUNING.md](FINE_TUNING.md) for the plan.

The model weights are CC BY-NC 4.0 (non-commercial); this package's code is MIT.
