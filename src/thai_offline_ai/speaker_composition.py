"""Deterministically align immutable ASR segments with speaker turns."""
from __future__ import annotations

from thai_offline_ai.asr_types import ASRResult
from thai_offline_ai.diarization_types import (
    DiarizationResult,
    SpeakerAttributedSegment,
    SpeakerAttributedTranscript,
)


def compose_speaker_transcript(
    asr: ASRResult, diarization: DiarizationResult
) -> SpeakerAttributedTranscript:
    """Assign greatest-overlap speaker and retain ambiguity instead of splitting text."""
    attributed: list[SpeakerAttributedSegment] = []
    for segment in asr.segments:
        metadata = dict(segment.metadata)
        speaker = None
        if segment.start is None or segment.end is None or segment.end <= segment.start:
            metadata["speaker_alignment"] = "missing_asr_timestamps"
        else:
            candidates = [
                (max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start)), turn)
                for turn in diarization.turns
            ]
            candidates = [(amount, turn) for amount, turn in candidates if amount > 0]
            if not candidates:
                metadata["speaker_alignment"] = "no_overlap"
            else:
                highest = max(amount for amount, _ in candidates)
                winners = [turn for amount, turn in candidates if amount == highest]
                winner = min(winners, key=lambda turn: (turn.start, turn.speaker))
                speaker = winner.speaker
                metadata["speaker_alignment"] = "greatest_overlap"
                if len(candidates) > 1:
                    metadata["speaker_alignment_ambiguous"] = True
                    metadata["overlapping_speakers"] = [turn.speaker for _, turn in candidates]
                if len(winners) > 1:
                    metadata["speaker_alignment_tie"] = True
        attributed.append(
            SpeakerAttributedSegment(
                text=segment.text,
                start=segment.start,
                end=segment.end,
                speaker=speaker,
                confidence=segment.confidence,
                metadata=metadata,
            )
        )
    return SpeakerAttributedTranscript(
        text=asr.text,
        segments=attributed,
        asr_backend=asr.backend,
        asr_model=asr.model,
        diarization_backend=diarization.backend,
        diarization_model=diarization.model,
        metadata={"alignment": "greatest_temporal_overlap"},
    )
