"""Pure-NumPy + ONNX Runtime adapter for Kokoro Thai (12 voices, 24 kHz, Apache-2.0).

Implements the exact 3-graph CPU ONNX serving path from ``kunato/wayu-kokoro-thai-v1``:
``prosody_fp32.onnx`` -> host gather -> ``curves_fp32.onnx`` -> float64 SineGen/STFT
-> ``decoder_fp32.onnx`` -> 24 kHz WAV. Requires no PyTorch or CUDA.
"""
from __future__ import annotations

import contextlib
import importlib.resources
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from thai_offline_ai.errors import WorkbenchError

KOKORO_MODEL_NAME = "kokoro_thai_onnx"
KOKORO_MODEL_ALIASES = frozenset(
    {
        "kokoro_thai_onnx",
        "kokoro_thai",
        "kokoro",
        "wayu-kokoro-thai-v1",
        "kunato/wayu-kokoro-thai-v1",
    }
)
REQUIRED_KOKORO_FILES = (
    "onnx_manifest.json",
    "prosody_fp32.onnx",
    "curves_fp32.onnx",
    "decoder_fp32.onnx",
    "source_params.npz",
    "styles.npz",
)

SAMPLE_RATE = 24000
HARMONICS = 9
UPSAMPLE_SCALE = 300
VOICED_THRESHOLD = 10.0
SINE_AMP = 0.1
NOISE_STD = 0.003
N_FFT = 20
HOP = 5
DEFAULT_SEED = 1234
MAX_CHUNK_TOKENS = 220

NORMALIZE_IPA = {"\u1d10": "\u0254"}  # U+1D10 'ᴐ' -> U+0254 'ɔ'
TONE_DIGIT_TO_TOKEN = {
    "1": "\u2192",  # mid -> →
    "2": "\u02e9",  # low -> ˩
    "3": "\u2198",  # falling -> ↘
    "4": "\u2197",  # high -> ↗
    "5": "\u2193",  # rising -> ↓
}
_TAG_RE = re.compile(r"<[^>]*>")
_REPEAT_MARK_RE = re.compile(r"([ะัาำิีึืุูเแโใไ็่้๊๋์])\1+")
_REPEAT_CONS_RE = re.compile(r"([ก-ฮ])\1{2,}")
_SEG_RE = re.compile(r"([\u0e00-\u0e7f]+)|([A-Za-z][A-Za-z'\- ]*[A-Za-z]|[A-Za-z])|(\s+)|(.)")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?\n])\s+|\s{2,}")
_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Crisp Kokoro IPA pronunciations for common tech/media acronyms in Thai-English speech.
ENGLISH_ACRONYM_IPA: dict[str, str] = {
    "AI": "ˈAˌI",
    "API": "ˌApˌiˈI",
    "ASR": "ˌAˌɛsˈɑɹ",
    "STT": "ˌɛstˌitˈi",
    "TTS": "tˌitˌiˈɛs",
    "LLM": "ˌɛlˌɛlˈɛm",
    "NLP": "ˌɛnˌɛlpˈi",
    "CPU": "sˌipˌijˈu",
    "GPU": "ʤˌipˌijˈu",
    "UI": "jˌuˈI",
    "UX": "jˌuˈɛks",
    "ONNX": "ˈɑnɪks",
    "WAV": "wˈAv",
    "PDF": "pˌidˌiˈɛf",
    "URL": "jˌuˌɑɹˈɛl",
    "SQL": "ˌɛskjˌuˈɛl",
    "RAG": "ɹˈæɡ",
    "JSON": "ʤˈAsən",
    "CSV": "sˌiˌɛsvˈi",
    "CLI": "sˌiˌɛlˈI",
    "SDK": "ˌɛsdˌikˈA",
    "RTF": "ˌɑɹtˌiˈɛf",
    "CER": "sˌiˌiˈɑɹ",
    "WER": "dˌʌbəljˌuˌiˈɑɹ",
    "RAM": "ɹˈæm",
    "VRAM": "vˌiɹˈæm",
}
_LETTER_IPA: dict[str, str] = {
    "A": "ˈA", "B": "bˈi", "C": "sˈi", "D": "dˈi", "E": "ˈi",
    "F": "ˈɛf", "G": "ʤˈi", "H": "ˈAʧ", "I": "ˈI", "J": "ʤˈA",
    "K": "kˈA", "L": "ˈɛl", "M": "ˈɛm", "N": "ˈɛn", "O": "ˈO",
    "P": "pˈi", "Q": "kjˈu", "R": "ˈɑɹ", "S": "ˈɛs", "T": "tˈi",
    "U": "jˈu", "V": "vˈi", "W": "dˈʌbəljˌu", "X": "ˈɛks", "Y": "wˈI",
    "Z": "zˈi",
}


@dataclass(frozen=True, slots=True)
class KokoroVoiceSpec:
    voice_id: str
    speaker_index: int
    label: str
    gender: str
    default_speed: float


KOKORO_VOICES: tuple[KokoroVoiceSpec, ...] = (
    KokoroVoiceSpec(
        "f_young_bright", 1, "สาวสดใส น่าฟัง กระชับ · แนะนำ (f_young_bright)", "female", 1.05
    ),
    KokoroVoiceSpec(
        "f_young_clear", 2, "สาวเสียงใส ชัดเจน ไพเราะ (f_young_clear)", "female", 1.02
    ),
    KokoroVoiceSpec(
        "f_teen_bright", 0, "วัยรุ่น สดใส กระฉับกระเฉง (f_teen_bright)", "female", 1.10
    ),
    KokoroVoiceSpec(
        "f_young_warm", 3, "สาวเสียงนุ่ม อบอุ่น ไพเราะ (f_young_warm)", "female", 0.98
    ),
    KokoroVoiceSpec(
        "f_mid_clear", 4, "ผู้หญิงมืออาชีพ ชัดถ้อยชัดคำ (f_mid_clear)", "female", 1.00
    ),
    KokoroVoiceSpec(
        "f_mid_warm", 5, "ผู้หญิงอบอุ่น นุ่มนวล (f_mid_warm)", "female", 0.98
    ),
    KokoroVoiceSpec(
        "f_elderly_soft", 6, "ผู้หญิงผู้ใหญ่ นุ่มนวล (f_elderly_soft)", "female", 0.95
    ),
    KokoroVoiceSpec(
        "f_elderly_low", 7, "ผู้หญิงผู้ใหญ่ เสียงทุ้ม (f_elderly_low)", "female", 0.95
    ),
    KokoroVoiceSpec(
        "m_teen_bright", 8, "ชายวัยรุ่น สดใส (m_teen_bright)", "male", 1.05
    ),
    KokoroVoiceSpec(
        "m_young_clear", 9, "ชายวัยทำงาน ชัดเจน (m_young_clear)", "male", 1.00
    ),
    KokoroVoiceSpec(
        "m_mid_warm", 10, "ชายเสียงอบอุ่น (m_mid_warm)", "male", 0.98
    ),
    KokoroVoiceSpec(
        "m_elderly_deep", 11, "ชายผู้ใหญ่ เสียงทุ้ม (m_elderly_deep)", "male", 0.95
    ),
)
KOKORO_VOICE_MAP: dict[str, KokoroVoiceSpec] = {item.voice_id: item for item in KOKORO_VOICES}
DEFAULT_KOKORO_VOICE = "f_young_bright"


def is_kokoro_model(model_name: str | None, backend: str | None = None) -> bool:
    if backend == "kokoro_onnx":
        return True
    if not model_name:
        return False
    return model_name.strip().lower() in KOKORO_MODEL_ALIASES


def kokoro_model_is_available(model_dir: str | Path) -> bool:
    root = Path(model_dir)
    return root.is_dir() and all(
        (root / name).is_file() and (root / name).stat().st_size > 0
        for name in REQUIRED_KOKORO_FILES
    )


def kokoro_voice_choices() -> list[tuple[str, str]]:
    return [(spec.label, spec.voice_id) for spec in KOKORO_VOICES]


def normalize_thai_for_kokoro(text: str) -> str:
    """Normalize Thai numbers, maiyamok, and chat elongations before tltk G2P."""
    cleaned = text.strip()
    with contextlib.suppress(ImportError, OSError, RuntimeError, ValueError):
        from pythaitts.preprocess import preprocess_text

        cleaned = preprocess_text(cleaned)
    cleaned = _REPEAT_MARK_RE.sub(lambda m: m.group(1), cleaned)
    cleaned = _REPEAT_CONS_RE.sub(lambda m: m.group(1), cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _thai_chunk_to_ipa(chunk: str, th2ipa_fn: Any) -> str:
    with contextlib.suppress(KeyError, IndexError, ValueError, TypeError, RuntimeError):
        ipa = str(th2ipa_fn(chunk) or "")
        if ipa.strip():
            return ipa
    # Word-level fallback so one awkward word does not drop its neighbours.
    try:
        from pythainlp.tokenize import word_tokenize

        words = [w for w in word_tokenize(chunk, engine="newmm") if w.strip()]
    except (ImportError, OSError, RuntimeError, ValueError):
        words = [chunk]
    parts: list[str] = []
    for word in words:
        with contextlib.suppress(KeyError, IndexError, ValueError, TypeError, RuntimeError):
            piece = str(th2ipa_fn(word) or "").strip()
            if piece:
                parts.append(piece)
    return " ".join(parts)


@lru_cache(maxsize=1)
def _load_english_lexicon() -> dict[str, str]:
    """Load Misaki's bundled US English Kokoro-IPA dictionaries offline (zero PyTorch/spaCy)."""
    lexicon: dict[str, str] = {}
    with contextlib.suppress(ImportError, FileNotFoundError, OSError, json.JSONDecodeError):
        data_root = importlib.resources.files("misaki.data")
        for filename in ("us_silver.json", "us_gold.json"):
            resource = data_root.joinpath(filename)
            if not resource.is_file():
                continue
            raw = json.loads(resource.read_text(encoding="utf-8"))
            for key, value in raw.items():
                if isinstance(value, str):
                    lexicon[key] = value
                elif isinstance(value, dict):
                    chosen = value.get("DEFAULT") or next(iter(value.values()), "")
                    if isinstance(chosen, str) and chosen:
                        lexicon[key] = chosen
    return lexicon


@lru_cache(maxsize=1)
def _load_espeak_fallback() -> Any | None:
    """Load offline espeak-ng English G2P fallback once for proper nouns / OOV words."""
    with contextlib.suppress(ImportError, OSError, RuntimeError):
        from misaki.espeak import EspeakFallback

        return EspeakFallback(british=False)
    return None


def english_run_to_kokoro_phonemes(
    text: str,
    vocab: dict[str, int],
    *,
    lexicon: dict[str, str] | None = None,
    espeak_fallback: Any | None = None,
) -> str:
    """Convert an English/Latin run into Kokoro's shared IPA vocabulary offline."""
    words = _EN_WORD_RE.findall(text)
    if not words:
        return ""
    en_lex = _load_english_lexicon() if lexicon is None else lexicon
    fb = _load_espeak_fallback() if espeak_fallback is None else espeak_fallback
    pieces: list[str] = []
    for word in words:
        raw_ipa = ""
        upper = word.upper()
        if word.isupper() and upper in ENGLISH_ACRONYM_IPA:
            raw_ipa = ENGLISH_ACRONYM_IPA[upper]
        elif word in en_lex:
            raw_ipa = en_lex[word]
        elif word.lower() in en_lex:
            raw_ipa = en_lex[word.lower()]
        elif fb is not None:
            with contextlib.suppress(RuntimeError, ValueError, TypeError, OSError):
                ps, _ = fb(SimpleNamespace(text=word))
                if ps:
                    raw_ipa = str(ps)
        if not raw_ipa and word.isupper() and 1 <= len(word) <= 5:
            raw_ipa = " ".join(_LETTER_IPA.get(ch, "") for ch in upper).strip()
        filtered = "".join(c for c in raw_ipa if c == " " or c in vocab).strip()
        if filtered:
            pieces.append(filtered)
    return re.sub(r" +", " ", " ".join(pieces)).strip()


def thai_text_to_kokoro_phonemes(
    text: str,
    vocab: dict[str, int],
    *,
    th2ipa_fn: Any | None = None,
    en_lexicon: dict[str, str] | None = None,
    en_fallback: Any | None = None,
) -> str:
    """Convert Thai, English, or mixed Thai-English text into Kokoro's shared IPA vocabulary."""
    if th2ipa_fn is None:
        try:
            import tltk.nlp as _tltk

            th2ipa_fn = _tltk.th2ipa
        except (ImportError, OSError) as exc:
            raise WorkbenchError(
                "Kokoro Thai TTS requires 'tltk' for Thai phonemization. "
                "Run: python -m pip install tltk"
            ) from exc

    normalized = normalize_thai_for_kokoro(text)
    if not normalized:
        return ""

    out_parts: list[str] = []
    for thai_run, latin_run, ws, other in _SEG_RE.findall(normalized):
        if thai_run:
            raw_ipa = " ".join(_thai_chunk_to_ipa(part, th2ipa_fn) for part in thai_run.split())
            stripped = _TAG_RE.sub(" ", raw_ipa)
            tokens: list[str] = []
            for ch in stripped:
                if ch.isdigit():
                    if tokens and tokens[-1] != " ":
                        tone = TONE_DIGIT_TO_TOKEN.get(ch)
                        if tone and tone in vocab:
                            tokens.append(tone)
                elif ch in ".+":
                    continue
                elif ch.isspace():
                    if tokens and tokens[-1] != " ":
                        tokens.append(" ")
                else:
                    mapped = NORMALIZE_IPA.get(ch, ch)
                    if mapped in vocab:
                        tokens.append(mapped)
            if tokens:
                out_parts.append("".join(tokens))
        elif latin_run:
            en_ipa = english_run_to_kokoro_phonemes(
                latin_run, vocab, lexicon=en_lexicon, espeak_fallback=en_fallback
            )
            if en_ipa:
                out_parts.append(en_ipa)
        elif ws:
            out_parts.append(" ")
        elif other and other in vocab:
            out_parts.append(other)
    return re.sub(r" +", " ", "".join(out_parts)).strip()


def _split_phoneme_chunks(phonemes: str, vocab: dict[str, int]) -> list[str]:
    """Split long phoneme strings into <= MAX_CHUNK_TOKENS chunks at word boundaries."""
    words = phonemes.split()
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = sum(1 for ch in word if ch in vocab)
        if current and current_tokens + word_tokens + 1 > MAX_CHUNK_TOKENS:
            chunks.append(" ".join(current))
            current = [word]
            current_tokens = word_tokens
        else:
            current.append(word)
            current_tokens += word_tokens + (1 if current_tokens else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _resample_linear(x: Any, scale: float) -> Any:
    """Linear resampling along axis 1 matching torch F.interpolate(align_corners=False)."""
    import numpy as np

    length = x.shape[1]
    out_length = max(1, int(length * scale))
    source = (np.arange(out_length, dtype=np.float64) + 0.5) / scale - 0.5
    np.clip(source, 0, length - 1, out=source)
    lower = np.floor(source).astype(np.int64)
    upper = np.minimum(lower + 1, length - 1)
    weight = (source - lower)[None, :, None]
    return x[:, lower] * (1.0 - weight) + x[:, upper] * weight


def _stft_magnitude_phase(audio: Any, window: Any) -> Any:
    import numpy as np

    padded = np.pad(audio, N_FFT // 2, mode="reflect")
    n_frames = (len(padded) - N_FFT) // HOP + 1
    frames = np.lib.stride_tricks.as_strided(
        padded, (n_frames, N_FFT), (padded.strides[0] * HOP, padded.strides[0])
    )
    spectrum = np.fft.rfft(frames * window, axis=1).T[None]
    return np.concatenate([np.abs(spectrum), np.angle(spectrum)], axis=1).astype(np.float32)


def harmonic_source(
    f0_curve: Any,
    source_weight: Any,
    source_bias: Any,
    window: Any,
    *,
    seed: int = DEFAULT_SEED,
) -> Any:
    """Float64 SineGen + SourceModuleHnNSF + analysis STFT matching Kokoro iSTFTNet."""
    import numpy as np

    rng = np.random.default_rng(seed)
    f0 = np.repeat(np.asarray(f0_curve, dtype=np.float64), UPSAMPLE_SCALE, axis=1)[..., None]
    n_samples = f0.shape[1]

    rad = f0 * np.arange(1, HARMONICS + 1, dtype=np.float64) / SAMPLE_RATE
    rad -= np.floor(rad)
    rand_ini = rng.random((1, HARMONICS))
    rand_ini[:, 0] = 0.0
    rad[:, 0, :] += rand_ini

    phase = np.cumsum(_resample_linear(rad, 1.0 / UPSAMPLE_SCALE), axis=1) * 2.0 * np.pi
    phase = _resample_linear(phase * UPSAMPLE_SCALE, UPSAMPLE_SCALE)

    voiced = (f0 > VOICED_THRESHOLD).astype(np.float64)
    amplitude = voiced * NOISE_STD + (1.0 - voiced) * SINE_AMP / 3.0
    noise = rng.standard_normal((1, n_samples, HARMONICS))
    waves = np.sin(phase) * SINE_AMP * voiced + amplitude * noise

    merged = np.tanh(waves @ source_weight.T + source_bias)
    return _stft_magnitude_phase(merged[0, :, 0].astype(np.float32), window)


class KokoroThaiONNXModel:
    """Lazy, cache-only 24 kHz multi-voice Thai Kokoro ONNX engine for CPU."""

    def __init__(self, model_dir: str | Path, *, intra_threads: int = 4) -> None:
        root = Path(model_dir)
        if not kokoro_model_is_available(root):
            raise WorkbenchError(
                "Kokoro Thai ONNX model is not available locally. "
                "Run: python scripts/setup_kokoro_tts_model.py"
            )
        try:
            import numpy as np
            import onnxruntime as ort
        except (ImportError, OSError) as exc:
            raise WorkbenchError(
                'ONNX Runtime is missing. Run: python -m pip install -e ".[tts,dev]"'
            ) from exc

        self.model_dir = root
        manifest = json.loads((root / "onnx_manifest.json").read_text(encoding="utf-8"))
        self.vocab: dict[str, int] = dict(manifest["vocab"])

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.log_severity_level = 3

        self.prosody = ort.InferenceSession(
            str(root / "prosody_fp32.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.curves = ort.InferenceSession(
            str(root / "curves_fp32.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.decoder = ort.InferenceSession(
            str(root / "decoder_fp32.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        with np.load(root / "source_params.npz") as source:
            self.source_weight = np.asarray(source["weight"], dtype=np.float64)
            self.source_bias = np.asarray(source["bias"], dtype=np.float64)
            self.window = np.asarray(source["window"], dtype=np.float32)

        with np.load(root / "styles.npz") as styles_data:
            self.styles: dict[int, Any] = {
                int(k): np.asarray(styles_data[k], dtype=np.float32).reshape(1, 256)
                for k in styles_data.files
            }

        self.voicepacks: dict[int, Any] = {}
        vp_path = root / "voicepacks.npz"
        if vp_path.is_file() and vp_path.stat().st_size > 0:
            with np.load(vp_path) as vp_data:
                for key in vp_data.files:
                    arr = np.asarray(vp_data[key], dtype=np.float32)
                    try:
                        idx = int(key)
                    except ValueError:
                        spec = KOKORO_VOICE_MAP.get(key)
                        if spec is None:
                            continue
                        idx = spec.speaker_index
                    self.voicepacks[idx] = arr

    def _style_for(self, speaker_index: int, token_count: int) -> Any:
        pack = self.voicepacks.get(speaker_index)
        if pack is not None and pack.ndim >= 2:
            clamped = max(0, min(token_count, pack.shape[0] - 1))
            return pack[clamped].reshape(1, 256)
        style = self.styles.get(speaker_index)
        if style is None:
            style = next(iter(self.styles.values()))
        return style

    def synth_phonemes(
        self,
        phonemes: str,
        *,
        speaker_index: int = 1,
        speed: float = 1.05,
        seed: int = DEFAULT_SEED,
    ) -> Any:
        import numpy as np

        ids = [i for i in (self.vocab.get(ch) for ch in phonemes) if i is not None]
        if not ids:
            return np.zeros(0, dtype=np.float32)

        ref_s = self._style_for(speaker_index, len(ids))
        style_dur, style_acou = ref_s[:, 128:], ref_s[:, :128]
        pred_dur, d, t_en = self.prosody.run(
            None,
            {
                "input_ids": np.array([[0, *ids, 0]], dtype=np.int64),
                "style_dur": style_dur,
                "speed": np.array([float(speed)], dtype=np.float32),
            },
        )
        index = np.repeat(np.arange(pred_dur.shape[0], dtype=np.int64), pred_dur)
        en = np.ascontiguousarray(d.transpose(0, 2, 1)[:, :, index])
        asr = np.ascontiguousarray(t_en[:, :, index])
        f0_curve, n_curve = self.curves.run(None, {"en": en, "style_dur": style_dur})
        har = harmonic_source(
            f0_curve, self.source_weight, self.source_bias, self.window, seed=seed
        )
        (audio,) = self.decoder.run(
            None,
            {
                "asr": asr,
                "f0_curve": f0_curve,
                "n_curve": n_curve,
                "style_acou": style_acou,
                "har": har,
            },
        )
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def synthesize(
        self,
        text: str,
        *,
        voice: str = DEFAULT_KOKORO_VOICE,
        speed: float | None = None,
        seed: int = DEFAULT_SEED,
    ) -> tuple[Any, int]:
        import numpy as np

        spec = KOKORO_VOICE_MAP.get(voice, KOKORO_VOICE_MAP[DEFAULT_KOKORO_VOICE])
        effective_speed = spec.default_speed if speed is None else float(speed)
        effective_speed = max(0.7, min(1.5, effective_speed))

        clauses = [part.strip() for part in _CLAUSE_SPLIT_RE.split(text.strip()) if part.strip()]
        if not clauses:
            raise WorkbenchError("TTS input must contain text.")

        waveforms: list[Any] = []
        pause = np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32)
        for clause in clauses:
            phonemes = thai_text_to_kokoro_phonemes(clause, self.vocab)
            if not phonemes:
                continue
            for chunk in _split_phoneme_chunks(phonemes, self.vocab):
                wav = self.synth_phonemes(
                    chunk,
                    speaker_index=spec.speaker_index,
                    speed=effective_speed,
                    seed=seed,
                )
                if wav.size > 0:
                    if waveforms:
                        waveforms.append(pause)
                    waveforms.append(wav)
        if not waveforms:
            raise WorkbenchError("Could not phonemize Thai text for Kokoro TTS.")
        combined = np.concatenate(waveforms).astype(np.float32, copy=False)
        peak = float(np.max(np.abs(combined)))
        if peak > 0.95:
            combined = combined * (0.95 / peak)
        return combined, SAMPLE_RATE
