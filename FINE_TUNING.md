# Fine-tuning plan (needs a GPU)

Everything in `src/muscriptor_plus` is post-processing and multi-pass decoding:
it improves the *output* of a frozen MuScriptor model and needs no GPU. The
changes below would improve the model itself, and are written down rather than
implemented because this environment has no GPU, no training data, and the
released `muscriptor` package ships inference only - it has no training-time
note-to-token encoder, which is the first thing any fine-tuning run needs.

## What is missing, concretely

The released package contains the tokenizer's *decode* direction
(`build_event_vocab`, `note_event2note`, `note_event2midi`) but not the *encode*
direction (`note_event2event`, referenced in a comment in `tokenizer/mt3.py` as
the training encoder). Fine-tuning needs to turn a ground-truth MIDI back into
the model's token stream, chunk by chunk, with the same tie prologue the decoder
expects. That encoder has to be reconstructed from the MT3 / YourMT3+
conventions before anything else.

## Priority 1: a velocity head

The highest-value change, and the smallest. MuScriptor already spends a
`velocity` token (0/1) per note, so the token stream has a slot for dynamics
that currently carries no information.

1. **Data**: any dataset with expressive MIDI performances and audio - MAESTRO
   and the piano subset of a large multitrack set are the usual starting points,
   since velocity is annotated there. Match the 5-second chunking and 16 kHz
   mel conditioning the model was trained with.
2. **Target**: keep the note tokens as they are and change only the velocity
   token's meaning: instead of {0, 1} emit a quantised velocity in, say, 32
   bins. This needs a new token range, so the vocabulary grows.
3. **Head**: add a small velocity head on the transformer's final hidden state,
   predicting the bin for each note token. Train it with the backbone frozen
   first (a few hours on one GPU), then fine-tune end-to-end at a low learning
   rate.
4. **Baseline**: the `velocity` module in this package is the reference the head
   must beat - it is already a decent estimator, so a learned head only earns
   its place if it beats it on held-out pieces, especially on polyphony where
   overlapping harmonics make the heuristic's job hard.

## Priority 2: hard-passage fine-tuning

The `consensus` module attacks dense and unclear passages at inference time.
The same material can train the model to be better at them directly.

1. **Mine hard examples**: run the model on a large corpus, keep the passages
   where the consensus of several decodings disagrees most. Those are the dense
   and unclear passages, already labelled by the model's own uncertainty.
2. **Weight the loss**: fine-tune with extra loss weight on those regions, on
   top of the ordinary training set so the easy material isn't forgotten.
3. **Evaluate** with note-level F1 on the hard set specifically, not just an
   aggregate, or the improvement will be invisible.

## What to run it on

One 24 GB GPU is enough for the `medium` variant (307M) with the backbone
frozen; the `large` variant (1.4B) wants 40-80 GB, or sharding. This repository
has neither, so the plan above is a specification, not a claim of results.
