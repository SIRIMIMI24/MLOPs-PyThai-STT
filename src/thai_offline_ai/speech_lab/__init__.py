"""Phase 6.5 — Speech AI Continuous Improvement Lab.

Provides strict dataset separation & leakage guards, local synthetic speech +
audio augmentation, controlled pseudo-labeling with quality filters, local
SQLite experiment tracking, and an anti-degradation promotion gate reusing the
unchanged Phase 3.7 ASREvaluator.
"""

from __future__ import annotations

from thai_offline_ai.speech_lab.datasets import (
    VALID_SPLITS,
    DatasetManifest,
    DatasetSample,
    compute_file_sha256,
    validate_no_leakage,
)
from thai_offline_ai.speech_lab.experiments import (
    ExperimentRecord,
    ExperimentRegistry,
    PromotionCriteria,
    PromotionDecision,
    export_training_bundle,
)
from thai_offline_ai.speech_lab.pseudo_label import (
    PseudoLabelFilterConfig,
    PseudoLabelingPipeline,
    PseudoLabelRecord,
    PseudoLabelReport,
)
from thai_offline_ai.speech_lab.synthetic import (
    AugmentationConfig,
    SyntheticSpeechGenerator,
    augment_waveform,
)

__all__ = [
    "VALID_SPLITS",
    "AugmentationConfig",
    "DatasetManifest",
    "DatasetSample",
    "ExperimentRecord",
    "ExperimentRegistry",
    "PromotionCriteria",
    "PromotionDecision",
    "PseudoLabelFilterConfig",
    "PseudoLabelRecord",
    "PseudoLabelReport",
    "PseudoLabelingPipeline",
    "SyntheticSpeechGenerator",
    "augment_waveform",
    "compute_file_sha256",
    "export_training_bundle",
    "validate_no_leakage",
]
