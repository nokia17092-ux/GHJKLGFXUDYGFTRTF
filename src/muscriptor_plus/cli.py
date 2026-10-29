"""Command-line entry point: ``muscriptor-plus transcribe audio.wav``.

A thin wrapper around :class:`muscriptor_plus.enhanced.EnhancedTranscriber`
that exposes the enhancement knobs. The stock ``muscriptor`` CLI remains the
way to run the unmodified pipeline; this one is for the enhanced output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils.midi import ExportOptions

from muscriptor_plus.cleanup import CleanupOptions
from muscriptor_plus.consensus import ConsensusOptions
from muscriptor_plus.enhanced import EnhancedTranscriber
from muscriptor_plus.humanize import HumanizeOptions
from muscriptor_plus.velocity import VelocityOptions

app = typer.Typer(add_completion=False, help="muscriptor-plus — enhanced music transcription")


@app.callback()
def _main() -> None:
    """Keeps ``transcribe`` a named subcommand rather than the collapsed default."""


@app.command()
def transcribe(
    audio_file: Annotated[Path, typer.Argument(help="Input audio file")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output MIDI path (default: alongside input)"),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model size, local path, or hf:// URL"),
    ] = "large",
    device: Annotated[
        str | None, typer.Option("--device", "-d", help="cpu / cuda / mps / auto")
    ] = None,
    # dynamics
    no_velocity: Annotated[
        bool, typer.Option("--no-velocity", help="Keep the stock flat velocity")
    ] = False,
    no_autocalibrate: Annotated[
        bool,
        typer.Option(
            "--no-autocalibrate",
            help="Use the fixed absolute loudness reference instead of the piece's own",
        ),
    ] = False,
    # live feel
    no_humanize: Annotated[
        bool, typer.Option("--no-humanize", help="Skip phrase shaping and micro-timing")
    ] = False,
    phrase_shape: Annotated[
        str, typer.Option("--phrase-shape", help="arch / crescendo / diminuendo / none")
    ] = "arch",
    phrase_depth: Annotated[
        float, typer.Option("--phrase-depth", help="Strength of the phrase dynamic arc")
    ] = 10.0,
    jitter: Annotated[
        float, typer.Option("--jitter", help="Micro-timing offset, in seconds")
    ] = 0.012,
    bpm: Annotated[
        float | None, typer.Option("--bpm", help="Tempo for beat accents")
    ] = None,
    beats_per_bar: Annotated[
        int | None, typer.Option("--beats-per-bar", help="Meter for accents")
    ] = None,
    # dense passages
    consensus: Annotated[
        bool,
        typer.Option(
            "--consensus",
            help="Multi-pass decoding for dense/unclear passages (Nx slower)",
        ),
    ] = False,
    consensus_passes: Annotated[
        int, typer.Option("--consensus-passes", help="Number of decoding passes")
    ] = 5,
    min_agreement: Annotated[
        float, typer.Option("--min-agreement", help="Fraction of passes that must agree")
    ] = 0.5,
    # artifact cleanup
    no_cleanup: Annotated[
        bool, typer.Option("--no-cleanup", help="Keep likely decoder artifacts")
    ] = False,
    # export
    quantize: Annotated[
        bool, typer.Option("--quantize", help="Snap notes to the beat grid")
    ] = False,
):
    """Transcribe an audio file to MIDI with dynamics, live feel and cleanup."""
    if not audio_file.exists():
        typer.echo(f"Error: file not found: {audio_file}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading model {model} …", err=True)
    tm = TranscriptionModel.load_model(weights_path=model, device=device)

    velocity_opts = VelocityOptions(auto_calibrate=not no_autocalibrate)
    humanize_opts = HumanizeOptions(
        phrase_shape=phrase_shape,
        phrase_depth=phrase_depth,
        timing_jitter_s=jitter,
        bpm=bpm,
        beats_per_bar=beats_per_bar,
    )
    consensus_opts = None
    if consensus:
        base = ConsensusOptions()
        consensus_opts = ConsensusOptions(
            passes=base.passes[:consensus_passes],
            min_agreement=min_agreement,
        )

    enh = EnhancedTranscriber(
        tm,
        velocity=velocity_opts,
        humanize=humanize_opts,
        consensus=consensus_opts,
        cleanup=CleanupOptions(),
    )

    typer.echo("Transcribing …", err=True)
    result = enh.transcribe(
        audio_file,
        export=ExportOptions(quantize=quantize),
        apply_velocity=not no_velocity,
        apply_humanize=not no_humanize,
        apply_cleanup=not no_cleanup,
    )

    dest = output or audio_file.with_suffix(".enhanced.mid")
    result.save_midi(dest)
    typer.echo(
        f"Wrote {dest} ({len(result.notes)} notes, {result.passes} pass(es), "
        f"velocity {min(result.velocities, default=0)}–{max(result.velocities, default=0)})",
        err=True,
    )


if __name__ == "__main__":
    app()
