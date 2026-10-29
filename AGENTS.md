# Repository notes

`muscriptor-plus` is a post-processing / multi-pass-decoding layer over
[Muscriptor](https://huggingface.co/MuScriptor). It never modifies model
weights.

## Environment

* Python 3.10+; a `.venv` at the repo root holds the dev install.
* `pip install -e .` installs the package plus `muscriptor` and `torch`.
* CPU-only in this workspace (no GPU). A `large` decode runs at roughly 0.15x
  real time on CPU, so model-touching work is slow - run it in the background
  and poll a log file.

## Layout

* `src/muscriptor_plus/velocity.py` - per-note dynamics from audio.
* `src/muscriptor_plus/humanize.py` - phrase arcs, accents, micro-timing.
* `src/muscriptor_plus/consensus.py` - multi-pass decode merge.
* `src/muscriptor_plus/cleanup.py` - artifact removal.
* `src/muscriptor_plus/midi_writer.py` - MIDI writer with per-note velocity.
* `src/muscriptor_plus/enhanced.py` - `EnhancedTranscriber`, the entry point.
* `tools/download_model.py` - fetch weights (gated repo, else public mirror).
* `tools/upload_to_hub.py` - publish to Hugging Face (needs `HF_TOKEN`).

## Commands

```bash
pytest                 # fast tests; the model ones are deselected
pytest -m slow         # real-model end-to-end tests (needs models/)
pyflakes src tests tools
```

## Things worth remembering

* MuScriptor's velocity token is **binary**; the stock MIDI writer emits a
  constant velocity. That is the whole reason `velocity.py` exists.
* `Muscriptor/muscriptor-large` is gated (CC BY-NC 4.0). The mirror
  `sonic-coder/muscriptor-large` carries byte-identical weights and is not
  gated; `tools/download_model.py` tries the official repo first.
* The released `muscriptor` package is inference-only: it has **no** training
  note-to-token encoder (`note_event2event`), which any fine-tuning run needs.
  See `FINE_TUNING.md`.
* STFT caveat that bit us once: a centered STFT frame reported at time `t`
  contains audio from `t - n_fft/2`, so a measurement window starting exactly
  at a note onset leaks the previous note. `velocity.py` offsets by half a
  frame; keep that when changing FFT parameters.
