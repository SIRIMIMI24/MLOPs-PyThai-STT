"""Pure-NumPy + ONNX Runtime adapter for iApp OpenThai2.0-ASR-Streaming-ONNX.

Implements the official ONNX contract from ``iapp/OpenThai2.0-ASR-Streaming-ONNX``:
- 16 kHz mono audio -> 128 log-mel filterbank features (25 ms Hann window,
  10 ms hop, pre-emphasis 0.97, no per-feature normalization).
- Cache-aware FastConformer encoder with language prompt index
  (``th-TH`` = 32, ``th-mix`` = 63).
- 2-layer LSTM RNN-T greedy decoder + joint network with official blank bias
  (-1.0 standard, -2.5 for clips shorter than 2.0 seconds) and
  ``max_symbols_per_step = 2``.
- Pause-based chunking for long audio files so silence pauses do not drop words.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from thai_offline_ai.errors import WorkbenchError

OPENTHAI_MODEL_NAME = "openthai2_asr"
OPENTHAI_MODEL_ALIASES = frozenset(
    {
        "openthai2_asr",
        "openthai2",
        "openthai2.0",
        "openthai2.0-asr-streaming",
        "openthai2.0-asr-streaming-onnx",
        "iapp/openthai2.0-asr-streaming",
        "iapp/openthai2.0-asr-streaming-onnx",
    }
)
PROMPT_INDICES: dict[str, int] = {
    "th-TH": 32,
    "th-mix": 63,
}
REQUIRED_FILES_BY_PRECISION: dict[str, tuple[str, ...]] = {
    "int8": ("encoder.int8.onnx", "encoder.int8.onnx.data", "decoder_joint.onnx", "tokens.txt"),
    "fp32": ("encoder.onnx", "encoder.onnx.data", "decoder_joint.onnx", "tokens.txt"),
}
SPECIAL_TOKENS = frozenset({"<unk>", "<s>", "</s>", "<pad>", "<bos>", "<eos>", "<blk>"})
_LANG_TAG_RE = re.compile(r"<(?:[a-z]{2}-[A-Z]{2}|\|[^|]+\|)>")

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 128
PREEMPH = 0.97
SUBSAMPLING_FACTOR = 8
FRAME_DURATION = (HOP_LENGTH * SUBSAMPLING_FACTOR) / float(SAMPLE_RATE)  # 0.08 s
STANDARD_BLANK_BIAS = -1.0
SHORT_CLIP_BLANK_BIAS = -2.5
SHORT_CLIP_THRESHOLD_SECONDS = 2.0
MAX_SYMBOLS_PER_STEP = 2

_MEL_FB_CACHE: dict[tuple[int, int, int], Any] = {}


def is_openthai_model(model_name: str | None, backend: str | None = None) -> bool:
    if backend == "openthai_onnx":
        return True
    if not model_name:
        return False
    return model_name.strip().lower() in OPENTHAI_MODEL_ALIASES


def openthai_model_is_available(model_dir: str | Path, precision: str = "int8") -> bool:
    root = Path(model_dir)
    if not root.is_dir():
        return False
    names = REQUIRED_FILES_BY_PRECISION.get(precision, REQUIRED_FILES_BY_PRECISION["int8"])
    return all((root / name).is_file() and (root / name).stat().st_size > 0 for name in names)


def resolve_available_precision(model_dir: str | Path, preferred: str = "int8") -> str | None:
    if openthai_model_is_available(model_dir, preferred):
        return preferred
    fallback = "fp32" if preferred == "int8" else "int8"
    if openthai_model_is_available(model_dir, fallback):
        return fallback
    return None


def _hz_to_mel_slaney(frequencies: Any) -> Any:
    import numpy as np

    f_min = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    freq_arr = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    mels = (freq_arr - f_min) / f_sp
    mask = freq_arr >= min_log_hz
    if np.any(mask):
        mels[mask] = min_log_mel + np.log(freq_arr[mask] / min_log_hz) / logstep
    return mels


def _mel_to_hz_slaney(mels: Any) -> Any:
    import numpy as np

    f_min = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    mel_arr = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    freqs = f_min + f_sp * mel_arr
    mask = mel_arr >= min_log_mel
    if np.any(mask):
        freqs[mask] = min_log_hz * np.exp(logstep * (mel_arr[mask] - min_log_mel))
    return freqs


def create_mel_filterbank(
    sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS
) -> Any:
    key = (sr, n_fft, n_mels)
    if key in _MEL_FB_CACHE:
        return _MEL_FB_CACHE[key]
    import numpy as np

    fft_freqs = np.linspace(0.0, sr / 2.0, int(1 + n_fft // 2), endpoint=True)
    mel_min = float(_hz_to_mel_slaney(0.0)[0])
    mel_max = float(_hz_to_mel_slaney(sr / 2.0)[0])
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz_slaney(mel_points)
    fdiff = np.diff(hz_points)
    ramps = np.subtract.outer(hz_points, fft_freqs)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, np.newaxis]
    fb = weights.astype(np.float32)
    _MEL_FB_CACHE[key] = fb
    return fb


def extract_log_mel_128(
    audio: Any,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[Any, int]:
    """Extract 128-channel unnormalized log-mel features matching OpenThai2.0 ONNX."""
    import numpy as np

    signal = np.asarray(audio, dtype=np.float32).flatten()
    if signal.size == 0:
        return np.zeros((1, N_MELS, 0), dtype=np.float32), 0
    if PREEMPH > 0:
        signal = np.concatenate((signal[:1], signal[1:] - PREEMPH * signal[:-1]))

    # Periodic Hann window matching torch.hann_window(400, periodic=False) centered in 512 FFT.
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(WIN_LENGTH, dtype=np.float64) / (WIN_LENGTH - 1)))
    padded_window = np.zeros(N_FFT, dtype=np.float64)
    start = (N_FFT - WIN_LENGTH) // 2
    padded_window[start : start + WIN_LENGTH] = window

    pad_amount = N_FFT // 2
    if len(signal) < pad_amount + 1:
        signal = np.pad(signal, (0, pad_amount + 1 - len(signal)), mode="constant")
    padded_signal = np.pad(signal, (pad_amount, pad_amount), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded_signal, window_shape=N_FFT)[::HOP_LENGTH]
    spec = np.fft.rfft(frames * padded_window, n=N_FFT, axis=1)
    power_spec = (np.abs(spec) ** 2).T.astype(np.float32)

    fb = create_mel_filterbank(sr=sample_rate, n_fft=N_FFT, n_mels=N_MELS)
    mel_spec = np.matmul(fb, power_spec)
    log_mel = np.log(mel_spec + float(2.0**-24)).astype(np.float32)
    seq_len = int(log_mel.shape[1])
    return np.expand_dims(log_mel, axis=0), seq_len


def split_audio_at_pauses(
    audio: Any,
    sample_rate: int = SAMPLE_RATE,
    *,
    max_chunk_seconds: float = 20.0,
    min_chunk_seconds: float = 6.0,
) -> list[tuple[float, Any]]:
    """Split long audio at low-energy pauses so long recordings keep high recall."""
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32).flatten()
    max_samples = round(max_chunk_seconds * sample_rate)
    min_samples = round(min_chunk_seconds * sample_rate)
    if len(samples) <= max_samples:
        return [(0.0, samples)]

    frame_size = round(0.05 * sample_rate)  # 50 ms RMS frames
    pieces: list[tuple[float, Any]] = []
    cursor = 0
    while cursor < len(samples):
        remaining = len(samples) - cursor
        if remaining <= max_samples:
            pieces.append((cursor / sample_rate, samples[cursor:]))
            break
        search_start = cursor + min_samples
        search_end = min(len(samples), cursor + max_samples)
        window = samples[search_start:search_end]
        num_frames = max(1, len(window) // frame_size)
        framed = window[: num_frames * frame_size].reshape(num_frames, frame_size)
        rms = np.sqrt(np.mean(framed**2, axis=1))
        best_frame = int(np.argmin(rms))
        split_at = search_start + best_frame * frame_size + frame_size // 2
        pieces.append((cursor / sample_rate, samples[cursor:split_at]))
        cursor = split_at
    return pieces


def _clean_piece(piece: str) -> str:
    return _LANG_TAG_RE.sub("", piece)


def _decode_token_ids(tokens: list[str], token_ids: list[int]) -> str:
    pieces: list[str] = []
    for tid in token_ids:
        if 0 <= tid < len(tokens):
            tok = tokens[tid]
            if tok in SPECIAL_TOKENS:
                continue
            cleaned = _clean_piece(tok)
            if cleaned:
                pieces.append(cleaned)
    return "".join(pieces).replace("\u2581", " ").strip()


def _tokens_to_chunks(
    tokens: list[str],
    token_ids: list[int],
    frame_indices: list[int],
    *,
    time_offset: float = 0.0,
) -> list[dict[str, Any]]:
    items: list[tuple[str, int]] = []
    for tid, frame_idx in zip(token_ids, frame_indices):
        if 0 <= tid < len(tokens):
            tok = tokens[tid]
            if tok in SPECIAL_TOKENS:
                continue
            cleaned = _clean_piece(tok)
            if cleaned:
                items.append((cleaned, frame_idx))
    if not items:
        return []
    chunks: list[dict[str, Any]] = []
    curr_tokens: list[str] = []
    curr_start: int | None = None
    curr_end: int | None = None
    for tok_str, frame_idx in items:
        is_new = (
            curr_start is None
            or tok_str.startswith(("\u2581", " "))
            or (curr_end is not None and (frame_idx - curr_end) >= 6)
        )
        if is_new and curr_tokens and curr_start is not None and curr_end is not None:
            text = "".join(curr_tokens).replace("\u2581", " ").strip()
            if text:
                start_s = round(time_offset + curr_start * FRAME_DURATION, 2)
                end_s = round(time_offset + (curr_end + 1) * FRAME_DURATION, 2)
                chunks.append(
                    {"text": text, "timestamp": (start_s, end_s), "start": start_s, "end": end_s}
                )
            curr_tokens = []
            curr_start = frame_idx
        if curr_start is None:
            curr_start = frame_idx
        curr_tokens.append(tok_str)
        curr_end = frame_idx
    if curr_tokens and curr_start is not None and curr_end is not None:
        text = "".join(curr_tokens).replace("\u2581", " ").strip()
        if text:
            start_s = round(time_offset + curr_start * FRAME_DURATION, 2)
            end_s = round(time_offset + (curr_end + 1) * FRAME_DURATION, 2)
            chunks.append(
                {"text": text, "timestamp": (start_s, end_s), "start": start_s, "end": end_s}
            )
    return chunks


class OpenThai2ONNXModel:
    """Lazy, cache-only ONNX Runtime wrapper for OpenThai2.0-ASR-Streaming."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        precision: str = "int8",
        device: str = "cpu",
        max_symbols_per_step: int = MAX_SYMBOLS_PER_STEP,
    ) -> None:
        resolved = resolve_available_precision(model_dir, precision)
        if resolved is None:
            raise WorkbenchError(
                "OpenThai2.0 ASR ONNX model is not available locally. "
                f"Run: python scripts/setup_openthai_asr_model.py --precision {precision}"
            )
        try:
            import onnxruntime as ort
        except (ImportError, OSError) as exc:
            raise WorkbenchError(
                'ONNX Runtime is missing. Run: python -m pip install -e ".[all]"'
            ) from exc

        self.model_dir = Path(model_dir)
        self.precision = resolved
        self.max_symbols_per_step = max_symbols_per_step

        encoder_file = "encoder.int8.onnx" if resolved == "int8" else "encoder.onnx"
        self.encoder_path = self.model_dir / encoder_file
        self.decoder_path = self.model_dir / "decoder_joint.onnx"
        self.tokens_path = self.model_dir / "tokens.txt"

        self.tokens = self.tokens_path.read_text(encoding="utf-8").splitlines()
        self.blank_id = len(self.tokens)

        providers = ["CPUExecutionProvider"]
        avail = ort.get_available_providers()
        if device in {"cuda", "auto"} and "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3

        self.encoder_session = ort.InferenceSession(
            str(self.encoder_path), sess_options=opts, providers=providers
        )
        self.decoder_session = ort.InferenceSession(
            str(self.decoder_path), sess_options=opts, providers=providers
        )

        # Inspect verified tensor names and shapes directly from the ONNX graphs.
        enc_inputs = {inp.name: inp for inp in self.encoder_session.get_inputs()}
        self._enc_audio_name = "audio_signal" if "audio_signal" in enc_inputs else next(iter(enc_inputs))
        self._enc_length_name = "length" if "length" in enc_inputs else list(enc_inputs)[1]
        self._enc_prompt_name = "prompt_index" if "prompt_index" in enc_inputs else (
            list(enc_inputs)[2] if len(enc_inputs) > 2 else None
        )
        self._enc_length_dtype = enc_inputs[self._enc_length_name].type
        self._enc_prompt_dtype = (
            enc_inputs[self._enc_prompt_name].type if self._enc_prompt_name else "tensor(int64)"
        )

        dec_inputs = self.decoder_session.get_inputs()
        self._dec_input_names = [inp.name for inp in dec_inputs]
        self._dec_target_dtype = dec_inputs[1].type if len(dec_inputs) > 1 else "tensor(int32)"
        self._dec_tlen_dtype = dec_inputs[2].type if len(dec_inputs) > 2 else "tensor(int32)"
        state_shape = dec_inputs[3].shape if len(dec_inputs) > 3 else [2, 1, 640]
        self.num_layers = int(state_shape[0]) if isinstance(state_shape[0], int) else 2
        self.hidden_dim = int(state_shape[2]) if isinstance(state_shape[2], int) else 640

    @staticmethod
    def _np_int_dtype(onnx_type: str) -> Any:
        import numpy as np

        return np.int32 if "int32" in onnx_type else np.int64

    def encode(self, features: Any, seq_len: int, prompt_mode: str = "th-TH") -> tuple[Any, int]:
        import numpy as np

        prompt_idx = PROMPT_INDICES.get(prompt_mode, PROMPT_INDICES["th-TH"])
        feeds: dict[str, Any] = {
            self._enc_audio_name: np.asarray(features, dtype=np.float32),
            self._enc_length_name: np.array([seq_len], dtype=self._np_int_dtype(self._enc_length_dtype)),
        }
        if self._enc_prompt_name is not None:
            feeds[self._enc_prompt_name] = np.array(
                [prompt_idx], dtype=self._np_int_dtype(self._enc_prompt_dtype)
            )
        outputs = self.encoder_session.run(None, feeds)
        encoded = np.asarray(outputs[0], dtype=np.float32)
        # Normalize encoder output layout to (1, hidden=1024, T_enc).
        if encoded.ndim == 3 and encoded.shape[1] != 1024 and encoded.shape[2] == 1024:
            encoded = np.transpose(encoded, (0, 2, 1))
        if len(outputs) > 1:
            enc_len = int(np.asarray(outputs[1]).reshape(-1)[0])
        else:
            enc_len = int(encoded.shape[2])
        return encoded, min(enc_len, int(encoded.shape[2]))

    def decode_step(
        self,
        enc_frame: Any,
        last_token: int,
        state_h: Any,
        state_c: Any,
    ) -> tuple[Any, Any, Any]:
        import numpy as np

        targets = np.array([[last_token]], dtype=self._np_int_dtype(self._dec_target_dtype))
        target_length = np.array([1], dtype=self._np_int_dtype(self._dec_tlen_dtype))
        feeds = {
            self._dec_input_names[0]: enc_frame,
            self._dec_input_names[1]: targets,
            self._dec_input_names[2]: target_length,
            self._dec_input_names[3]: state_h,
            self._dec_input_names[4]: state_c,
        }
        outputs = self.decoder_session.run(None, feeds)
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        # Outputs are either (logits, length, h, c) or (logits, h, c).
        if len(outputs) >= 4:
            next_h, next_c = outputs[2], outputs[3]
        else:
            next_h, next_c = outputs[1], outputs[2]
        return logits, next_h, next_c

    def greedy_decode(
        self,
        encoded: Any,
        enc_len: int,
        *,
        blank_bias: float = STANDARD_BLANK_BIAS,
        state_h: Any | None = None,
        state_c: Any | None = None,
        last_token: int | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> tuple[list[int], list[int], Any, Any, int]:
        import numpy as np

        h = (
            state_h
            if state_h is not None
            else np.zeros((self.num_layers, 1, self.hidden_dim), dtype=np.float32)
        )
        c = (
            state_c
            if state_c is not None
            else np.zeros((self.num_layers, 1, self.hidden_dim), dtype=np.float32)
        )
        token = self.blank_id if last_token is None else last_token
        emitted_tokens: list[int] = []
        emitted_frames: list[int] = []
        stop_frame = min(enc_len, enc_len if end_frame is None else end_frame)

        for t in range(max(0, start_frame), stop_frame):
            enc_frame = encoded[:, :, t : t + 1]
            symbols_added = 0
            while symbols_added < self.max_symbols_per_step:
                logits, next_h, next_c = self.decode_step(enc_frame, token, h, c)
                if blank_bias != 0.0 and self.blank_id < len(logits):
                    logits = logits.copy()
                    logits[self.blank_id] += blank_bias
                pred = int(np.argmax(logits))
                if pred == self.blank_id:
                    break
                emitted_tokens.append(pred)
                emitted_frames.append(t)
                token = pred
                h = next_h
                c = next_c
                symbols_added += 1
        return emitted_tokens, emitted_frames, h, c, token

    def transcribe(
        self,
        audio: Any,
        sample_rate: int = SAMPLE_RATE,
        *,
        prompt_mode: str = "th-TH",
        return_timestamps: bool = True,
    ) -> dict[str, Any]:
        import numpy as np

        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)
        if samples.size == 0:
            return {"text": "", "chunks": []}

        pieces = split_audio_at_pauses(samples, sample_rate=sample_rate)
        texts: list[str] = []
        all_chunks: list[dict[str, Any]] = []
        for offset_s, chunk_audio in pieces:
            clip_duration = len(chunk_audio) / float(sample_rate)
            blank_bias = (
                SHORT_CLIP_BLANK_BIAS
                if clip_duration < SHORT_CLIP_THRESHOLD_SECONDS
                else STANDARD_BLANK_BIAS
            )
            features, seq_len = extract_log_mel_128(chunk_audio, sample_rate=sample_rate)
            if seq_len == 0:
                continue
            encoded, enc_len = self.encode(features, seq_len, prompt_mode=prompt_mode)
            token_ids, frame_ids, _, _, _ = self.greedy_decode(
                encoded, enc_len, blank_bias=blank_bias
            )
            text = _decode_token_ids(self.tokens, token_ids)
            if text:
                texts.append(text)
            if return_timestamps:
                all_chunks.extend(
                    _tokens_to_chunks(self.tokens, token_ids, frame_ids, time_offset=offset_s)
                )
        return {"text": " ".join(texts).strip(), "chunks": all_chunks}


class OpenThai2RealtimeStreamASR:
    """Sliding-context streaming decoder matching StreamingASRService's decoder contract."""

    def __init__(
        self,
        model: OpenThai2ONNXModel,
        *,
        sample_rate: int = SAMPLE_RATE,
        step_sec: float = 1.0,
        left_context_sec: float = 1.12,
        right_context_sec: float = 0.56,
        prompt_mode: str = "th-TH",
    ) -> None:
        import numpy as np

        self.model = model
        self.sample_rate = sample_rate
        self.step_sec = step_sec
        self.left_context_sec = left_context_sec
        self.right_context_sec = right_context_sec
        self.prompt_mode = prompt_mode

        self.step_samples = max(1, int(sample_rate * step_sec))
        self.left_samples = max(0, int(sample_rate * left_context_sec))
        self.right_samples = max(0, int(sample_rate * right_context_sec))

        self.h = np.zeros((model.num_layers, 1, model.hidden_dim), dtype=np.float32)
        self.c = np.zeros((model.num_layers, 1, model.hidden_dim), dtype=np.float32)
        self.last_token = model.blank_id
        self.audio_history = np.zeros(0, dtype=np.float32)
        self.cursor_sample = 0

    def process_chunk(self, chunk: Any) -> list[str]:
        import numpy as np

        data = np.asarray(chunk, dtype=np.float32)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        self.audio_history = np.concatenate([self.audio_history, data])

        emitted: list[str] = []
        while (len(self.audio_history) - self.cursor_sample) >= (
            self.step_samples + self.right_samples
        ):
            text = self._decode_window(is_flush=False)
            if text:
                emitted.append(text)

        max_history = self.left_samples + self.sample_rate * 3
        if self.cursor_sample > max_history:
            trim = self.cursor_sample - self.left_samples
            self.audio_history = self.audio_history[trim:]
            self.cursor_sample -= trim
        return emitted

    def flush(self) -> str:
        emitted: list[str] = []
        while self.cursor_sample < len(self.audio_history):
            text = self._decode_window(is_flush=True)
            if text:
                emitted.append(text)
        return " ".join(emitted).strip()

    def _decode_window(self, *, is_flush: bool) -> str:
        cursor = self.cursor_sample
        total_len = len(self.audio_history)
        if cursor >= total_len:
            return ""
        win_start = max(0, cursor - self.left_samples)
        if is_flush:
            win_end = total_len
            step_end = total_len
        else:
            win_end = min(total_len, cursor + self.step_samples + self.right_samples)
            step_end = min(total_len, cursor + self.step_samples)
        window = self.audio_history[win_start:win_end]
        curr_start_sample = cursor - win_start
        curr_end_sample = step_end - win_start
        self.cursor_sample = step_end

        features, seq_len = extract_log_mel_128(window, sample_rate=self.sample_rate)
        if seq_len == 0:
            return ""
        encoded, enc_len = self.model.encode(features, seq_len, prompt_mode=self.prompt_mode)
        start_frame = round(curr_start_sample / float(HOP_LENGTH * SUBSAMPLING_FACTOR))
        end_frame = min(
            enc_len,
            max(start_frame + 1, round(curr_end_sample / float(HOP_LENGTH * SUBSAMPLING_FACTOR))),
        )
        token_ids, _, next_h, next_c, next_token = self.model.greedy_decode(
            encoded,
            enc_len,
            blank_bias=STANDARD_BLANK_BIAS,
            state_h=self.h,
            state_c=self.c,
            last_token=self.last_token,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        self.h = next_h
        self.c = next_c
        self.last_token = next_token
        return _decode_token_ids(self.model.tokens, token_ids)
