"""muscriptor-plus: quality and musicality upgrades for MuScriptor.

Three independent improvements that sit on top of the stock MuScriptor
transcription pipeline:

* :mod:`muscriptor_plus.velocity` - estimate real note dynamics (forte /
  piano) from the recording. MuScriptor only emits a binary velocity token, so
  every note is written to MIDI at the same loudness; this restores the
  performance's dynamic shape.
* :mod:`muscriptor_plus.humanize` - phrase-level velocity shaping and optional
  micro-timing so the transcription keeps a live feel instead of a flat grid.
* :mod:`muscriptor_plus.consensus` - decode a passage several ways and merge
  the results, which recovers notes and removes spurious ones in dense or
  unclear passages.

:class:`muscriptor_plus.enhanced.EnhancedTranscriber` wires all three together
behind the same call shape as ``muscriptor.TranscriptionModel``.
"""

from muscriptor_plus.cleanup import CleanupOptions, cleanup
from muscriptor_plus.consensus import ConsensusDecoder, ConsensusOptions
from muscriptor_plus.enhanced import EnhancedTranscriber, EnhancedResult
from muscriptor_plus.humanize import HumanizeOptions, Humanizer
from muscriptor_plus.midi_writer import VelNote, notes_to_midi
from muscriptor_plus.velocity import VelocityEstimator, VelocityOptions

__all__ = [
    "CleanupOptions",
    "ConsensusDecoder",
    "ConsensusOptions",
    "EnhancedTranscriber",
    "EnhancedResult",
    "HumanizeOptions",
    "Humanizer",
    "VelNote",
    "VelocityEstimator",
    "VelocityOptions",
    "cleanup",
    "notes_to_midi",
]

__version__ = "0.1.0"
