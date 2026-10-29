---
license: mit
library_name: muscriptor
tags:
- music
- music-transcription
- automatic-music-transcription
- amt
- audio-to-midi
- midi
- dynamics
- velocity
- music-information-retrieval
---

# muscriptor-plus

Enhancements for [MuScriptor](https://huggingface.co/MuScriptor) music
transcription: **real note dynamics**, a **live feel**, and **cleaner handling
of dense or unclear passages**.

This repository hosts the `muscriptor-plus` code (MIT). It does not ship model
weights: it runs on top of a stock MuScriptor checkpoint, whose weights are
CC BY-NC 4.0 and must be obtained separately.

## What it adds

MuScriptor's tokenizer has a binary velocity token, so its MIDI output is
dynamically flat - every note at the same loudness. `muscriptor-plus` restores
the performance's dynamics by measuring each note's own energy in the recording,
and adds three further improvements:

* **`velocity`** - per-note forte/piano from the audio, on an absolute dB scale
  so quiet pieces stay quiet.
* **`humanize`** - phrase-level dynamic arcs, beat accents and deterministic
  micro-timing for a live feel.
* **`consensus`** - multi-pass decoding that keeps only notes a majority of
  passes agree on, for dense or unclear passages.
* **`cleanup`** - removes likely artifacts: decay-tail blips, restrikes,
  same-pitch blips, and notes far quieter than their instrument's median.

## Use

```bash
pip install muscriptor-plus
```

```python
from muscriptor.transcription_model import TranscriptionModel
from muscriptor_plus import EnhancedTranscriber

model = TranscriptionModel.load_model("large")
enh = EnhancedTranscriber(model)
enh.transcribe("song.wav").save_midi("song.mid")
```

## Limits

This is post-processing plus multi-pass decoding over a frozen model. It cannot
recover notes the model never emitted and does not change instrument
classification. Weight-level improvements (a learned velocity head, fine-tuning
on hard passages) need a GPU and training data; see `FINE_TUNING.md`.

## Licence

Code: MIT. MuScriptor weights: CC BY-NC 4.0 (non-commercial). MuScriptor is a
collaboration between Mirelo and Kyutai; please cite their paper if you use it.

_This model card and the accompanying code were produced by an AI agent
(OpenHands) on behalf of the repository owner._
