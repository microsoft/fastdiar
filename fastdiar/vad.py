"""Causal streaming voice-activity detection.

silero-VAD scores fixed 32 ms frames; a causal hysteresis state machine then
smooths those per-frame probabilities into confirmed speech intervals, mirroring
the offline ``get_speech_timestamps`` heuristic (onset/offset thresholds,
minimum speech/silence durations and boundary padding) under a strict
left-to-right constraint.

The class is fed arbitrary audio chunks and emits *confirmed* speech intervals
(absolute sample ranges, monotonically increasing and non-overlapping). Silence
is dropped: only speech is ever reported, matching the offline ``_concat_speech``
behaviour. Confirmation is necessarily delayed by up to ``min_silence``/
``min_speech`` (~250 ms) so a closing/short segment can still be revised.

Two feeding regimes are supported:

  * **caller-paced** (default, ``shift_ms=None``): frames are scored as soon as
    the chunks handed to :meth:`StreamingVAD.__call__` complete them.
  * **fixed-rate** (``shift_ms`` given): the VAD updates on a regular
    ``shift_ms`` grid regardless of how the caller chunks the audio -- at every
    step all silero frames that are complete at the step boundary are scored.

In both regimes each frame is scored exactly once and the model's recurrent
state is never reset mid-stream, so the acoustic context is the whole past
stream and the emitted intervals are identical; ``shift_ms`` only controls
*when* results become available (the online update granularity), not what they
are. The frames completed by an update are scored as one batch
(:class:`_BatchedSilero`), so a long update (e.g. the diarizer's 60 s step on a
GPU) costs far less than as many single-frame calls.
"""

import torch
from silero_vad import load_silero_vad

# silero operates on exactly 512-sample frames at 16 kHz (256 at 8 kHz).
_FRAME_BY_SR = {16000: 512, 8000: 256}


class _BatchedSilero:
    """silero-VAD over consecutive frames of one stream, many frames per call.

    silero scores a frame, preceded by the last ``context`` samples of the
    stream, with an STFT and a conv encoder, then one step of an LSTM whose
    state carries the whole past. Only that step is sequential: the frames of a
    call go through the STFT and the encoder as one batch and through the LSTM
    (its weights, as a :class:`torch.nn.LSTM`) as one sequence. The
    probabilities are silero's frame-by-frame ones, up to rounding.
    """

    def __init__(self, sr: int) -> None:
        jit = load_silero_vad(onnx=False)
        self.net = jit._model if sr == 16000 else jit._model_8k
        self.context = self.net.context_size_samples
        cell = self.net.decoder.rnn
        self.lstm = torch.nn.LSTM(cell.weight_ih.shape[1], cell.weight_hh.shape[1])
        with torch.no_grad():
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                getattr(self.lstm, f"{name}_l0").copy_(getattr(cell, name))
        self.reset()

    def reset(self) -> None:
        self._state = None  # LSTM (h, c): zeros at the start of a stream
        self._context = torch.zeros(self.context)

    @torch.inference_mode()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Speech probability of each of the ``(n, frame)`` consecutive frames."""
        n, size = frames.shape
        stream = torch.cat([self._context, frames.reshape(-1)])
        self._context = stream[-self.context :]
        x = stream.unfold(0, self.context + size, size)  # (n, context + frame)
        feats = self.net.encoder(self.net.stft(x)).squeeze(-1)  # (n, hidden)
        h, self._state = self.lstm(feats.unsqueeze(1), self._state)
        return self.net.decoder.decoder(h.squeeze(1).unsqueeze(-1)).reshape(n)


class StreamingVAD:
    """Streaming silero-VAD with causal hysteresis smoothing.

    Args:
        sr: Sample rate of the incoming audio (16 kHz or 8 kHz).
        threshold: Speech onset probability (silero default 0.5).
        neg_threshold: Speech offset probability; defaults to
            ``threshold - 0.15`` (silero's internal hysteresis).
        min_speech_ms: Discard speech runs shorter than this (smoothing window).
        min_silence_ms: Require this much silence before closing a segment
            (hang-over / smoothing window).
        speech_pad_ms: Pad each confirmed segment by this much on both sides.
        shift_ms: Update the VAD on a fixed ``shift_ms`` grid instead of
            whenever the caller's chunks complete a frame. The recurrent state
            is kept (unlimited past context), so this only changes the update
            granularity, not the emitted intervals. ``None`` (default) scores
            frames as the incoming chunks complete them.
    """

    def __init__(
        self,
        *,
        sr: int = 16000,
        threshold: float = 0.5,
        neg_threshold: float | None = None,
        min_speech_ms: int = 250,
        min_silence_ms: int = 250,
        speech_pad_ms: int = 30,
        shift_ms: int | None = None,
    ) -> None:
        if sr not in _FRAME_BY_SR:
            raise ValueError(f"silero VAD supports sr in {list(_FRAME_BY_SR)}, got {sr}")
        if shift_ms is not None and shift_ms <= 0:
            raise ValueError(f"shift_ms must be positive, got {shift_ms}")
        self.sr = sr
        self.frame = _FRAME_BY_SR[sr]
        self.threshold = threshold
        self.neg_threshold = threshold - 0.15 if neg_threshold is None else neg_threshold
        self.min_speech = max(1, round(min_speech_ms * sr / 1000 / self.frame))
        self.min_silence = max(1, round(min_silence_ms * sr / 1000 / self.frame))
        self.pad = int(round(speech_pad_ms * sr / 1000))
        self.shift = None if shift_ms is None else max(1, int(round(shift_ms * sr / 1000)))

        self.model = _BatchedSilero(sr)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all state for a new audio stream (call once per file)."""
        self.model.reset()
        self._tail = torch.zeros(0)  # leftover samples (< one frame)
        self._frame_idx = 0  # index of the next frame to be scored
        self._triggered = False  # inside a (possibly unconfirmed) speech run
        self._seg_start = 0  # frame index where the current run started
        self._confirmed = False  # current run already passed min_speech
        self._temp_end: int | None = None  # frame where a pending silence began
        self._emit_end = 0  # absolute sample up to which we have emitted
        # Fixed-rate regime only: audio not yet consumed by a step, plus the
        # step / stream counters.
        self._buf = torch.zeros(0)
        self._buf_start = 0  # absolute sample index of ``self._buf[0]``
        self._n_seen = 0  # total samples received so far
        self._step_idx = 0  # number of completed ``shift`` steps

    # ------------------------------------------------------------------
    def __call__(self, audio_chunk: torch.Tensor) -> list[tuple[int, int]]:
        """Process one audio chunk and return newly confirmed speech intervals.

        Args:
            audio_chunk: 1-D float tensor of samples at ``self.sr``.

        Returns:
            List of ``(start_sample, end_sample)`` absolute sample intervals,
            monotonically increasing and non-overlapping. May be empty.
        """
        x = audio_chunk.reshape(-1).float()
        if self.shift is not None:
            return self._call_stepped(x)

        x = torch.cat([self._tail, x])
        n_frames = x.shape[0] // self.frame
        out: list[tuple[int, int]] = []
        for prob in self._score(x[: n_frames * self.frame]):
            out.extend(self._step(prob))
        self._tail = x[n_frames * self.frame :]
        return out

    def flush(self) -> list[tuple[int, int]]:
        """Close any open confirmed segment at end-of-stream."""
        out: list[tuple[int, int]] = []
        if self.shift is not None:
            # Score the trailing (shorter than ``shift``) step so no complete
            # frame is left unscored at end-of-stream.
            out.extend(self._score_up_to(self._n_seen))
        if self._triggered and self._confirmed:
            out.extend(self._emit(self._frame_idx, closing=True))
        self._triggered = False
        self._confirmed = False
        self._temp_end = None
        return out

    def decided_upto(self) -> int:
        """Absolute sample up to which speech/silence can no longer be revised.

        Everything below the returned sample that was not reported by
        :meth:`__call__` is final silence. Callers that need a per-frame
        speech mask (rather than speech-only audio) must wait for this
        frontier before labelling a frame.
        """
        if self._triggered:
            if self._confirmed:
                return self._emit_end
            # An unconfirmed run may still be emitted from its padded onset.
            return max(self._emit_end, self._seg_start * self.frame - self.pad)
        # A run opening at the next frame would pad back by ``self.pad``.
        return max(self._emit_end, self._frame_idx * self.frame - self.pad)

    # ------------------------------------------------------------------
    def _call_stepped(self, x: torch.Tensor) -> list[tuple[int, int]]:
        """Fixed-rate feeding: run one update per completed ``shift`` step."""
        self._buf = torch.cat([self._buf, x])
        self._n_seen += x.shape[0]
        out: list[tuple[int, int]] = []
        while (self._step_idx + 1) * self.shift <= self._n_seen:
            self._step_idx += 1
            out.extend(self._score_up_to(self._step_idx * self.shift))
        return out

    def _score_up_to(self, end_sample: int) -> list[tuple[int, int]]:
        """Score every not-yet-scored frame that is complete at ``end_sample``.

        Only the newly arrived frames are fed to silero -- already scored audio
        is never re-submitted -- and the model's recurrent state is carried over
        from the previous step, so each frame keeps the full past stream as
        context.
        """
        f_end = end_sample // self.frame
        start = self._frame_idx * self.frame - self._buf_start
        out: list[tuple[int, int]] = []
        for prob in self._score(self._buf[start : f_end * self.frame - self._buf_start]):
            out.extend(self._step(prob))

        # Consumed samples are never needed again.
        drop = f_end * self.frame - self._buf_start
        if drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start += drop
        return out

    def _score(self, samples: torch.Tensor) -> list[float]:
        """Speech probabilities of the whole frames of ``samples``, scored as one batch."""
        n_frames = samples.shape[0] // self.frame
        if n_frames <= 0:
            return []
        return self.model(samples[: n_frames * self.frame].reshape(n_frames, self.frame)).tolist()

    # ------------------------------------------------------------------
    def _step(self, prob: float) -> list[tuple[int, int]]:
        """Advance the hysteresis state machine by one scored frame."""
        i = self._frame_idx
        out: list[tuple[int, int]] = []

        if prob >= self.threshold:
            self._temp_end = None  # speech (re)active: cancel any pending end
            if not self._triggered:
                self._triggered = True
                self._seg_start = i
                self._confirmed = False
        elif self._triggered and prob < self.neg_threshold:
            if self._temp_end is None:
                self._temp_end = i
            if i - self._temp_end >= self.min_silence:
                # Enough trailing silence: close the run at ``temp_end``.
                if self._confirmed:
                    out.extend(self._emit(self._temp_end, closing=True))
                self._triggered = False
                self._confirmed = False
                self._temp_end = None

        # Confirm the onset once the run is long enough to be kept.
        if self._triggered and not self._confirmed and (i - self._seg_start + 1) >= self.min_speech:
            self._confirmed = True

        # Emit the part of a confirmed run that can no longer be revised.
        if self._triggered and self._confirmed:
            safe = i + 1 if self._temp_end is None else self._temp_end
            out.extend(self._emit(safe, closing=False))

        self._frame_idx = i + 1
        return out

    def _emit(self, safe_frame: int, *, closing: bool) -> list[tuple[int, int]]:
        """Emit confirmed speech samples up to ``safe_frame`` (exclusive).

        Start padding is applied once at the run's onset; end padding only when
        the run actually closes. Overlaps with previously emitted samples are
        clipped, keeping the output monotonic and gap-free within speech.
        """
        start = self._seg_start * self.frame - self.pad
        end = safe_frame * self.frame + (self.pad if closing else 0)
        start = max(start, self._emit_end, 0)
        if end <= start:
            return []
        self._emit_end = end
        return [(start, end)]
