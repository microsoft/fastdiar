"""End-to-end streaming speaker diarization.

Chains the three causal stages over the audio stream::

    audio chunk (shift_sec)
      -> StreamingVAD             -> speech/silence mask of every encoder frame
      -> StreamingEncoder         -> one embedding per 80 ms frame (silence included)
      -> OnlineClustering         -> per-frame speaker id, ``max_delay_sec`` behind
      -> contiguous-label merge  -> RTTM lines of finalized speaker turns

Both stages see the same audio at every step: the encoder is fed the raw stream
and the VAD only says which frames are speech, so no time remapping is
involved. Latency is the VAD confirmation delay (~250 ms, the frontier reported
by :meth:`StreamingVAD.decided_upto`) plus ``max_delay_sec``; ``shift_sec`` only
affects how often results are produced, never their value.

Turns are written in the RTTM format of pyannote (``Annotation.write_rttm``)::

    SPEAKER <uri> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
"""

from collections import deque
from collections.abc import Callable, Iterator

import numpy as np
import torch

from fastdiar.clustering import OnlineClustering
from fastdiar.encoder import StreamingEncoder, StreamingReDimNet2, default_shift_sec
from fastdiar.vad import StreamingVAD

SAMPLE_RATE = 16000


class StreamingDiarizer:
    """Causal streaming diarizer producing RTTM.

    Args:
        model: A streaming model (see :func:`fastdiar.encoder.load_streaming_model`).
        shift_sec: Streaming step: the encoder block and, unless overridden in
            ``vad_kwargs``, the VAD update grid (latency/throughput knob). It
            sets how often results are produced, never their value. Default:
            the fastest for the model's device (:func:`default_shift_sec`),
            60 s on a GPU and 0.32 s on the CPU; pass 0.32 for a live stream
            on a GPU.
        delay_frames: ``OnlineClustering`` confidence look-back in frames.
        clust_th: Cluster assignment similarity threshold.
        sub_clust_th: Speech-cluster merging similarity threshold.
        confidence: Look-back similarity required to assign a frame.
        online_merge: Enable streaming speech-cluster merging.
        post_process: Offline speech-cluster merging. The whole stream is
            relabeled at end-of-file, so nothing is emitted incrementally and the
            final lines carry every turn.
        max_delay_sec: Fixed clustering output lag, in seconds. It doubles as
            the self-correction window: a frame is emitted with the label it
            holds when it leaves the window, not the one it first got.
        vad_kwargs: Extra keyword args forwarded to :class:`StreamingVAD`;
            ``shift_ms`` defaults to ``shift_sec`` so the VAD and the encoder
            consume the same audio at every step.
        on_frame: Called as ``on_frame(frame, speaker)`` for every frame as its
            label is finalized, in frame order (``speaker`` is ``None`` for
            silence), e.g. to display the labels live.

    The encoder runs on the model's device; the VAD and the clustering on the CPU.
    Speakers are numbered from 1 in the order they first speak in the output;
    clusters that never emit speech (e.g. merged away first) take no number.
    """

    def __init__(
        self,
        model: StreamingReDimNet2,
        *,
        shift_sec: float | None = None,
        delay_frames: int = 10,
        clust_th: float = 0.4,
        sub_clust_th: float = 0.8,
        confidence: float = 0.8,
        online_merge: bool = True,
        post_process: bool = False,
        max_delay_sec: float = 0.96,
        vad_kwargs: dict | None = None,
        on_frame: Callable[[int, int | None], None] | None = None,
    ) -> None:
        self.post_process = post_process
        self.on_frame = on_frame
        if shift_sec is None:
            shift_sec = default_shift_sec(model)
        vad_kwargs = {"shift_ms": round(shift_sec * 1000), **(vad_kwargs or {})}
        self.vad = StreamingVAD(sr=SAMPLE_RATE, **vad_kwargs)
        # The encoder block is a whole number of backbone frames.
        self.frame_samples = StreamingEncoder(model, shift_frames=1).frame_samples
        shift_frames = max(1, round(shift_sec * SAMPLE_RATE / self.frame_samples))
        self.encoder = StreamingEncoder(model, shift_frames=shift_frames)
        self.frame_shift = self.frame_samples / SAMPLE_RATE

        self._online_kwargs = {
            "delay_frames": delay_frames,
            "clust_th": clust_th,
            "sub_clust_th": sub_clust_th,
            "confidence": confidence,
            "sec_per_frame": self.frame_shift,
            "online_merge": online_merge,
            "post_process": post_process,
            "max_delay_sec": max_delay_sec,
        }
        # Outer feed granularity == encoder block (ties yield cadence to shift).
        self.input_chunk = self.encoder.block
        self.reset()

    def reset(self) -> None:
        """Reset every stage and all bookkeeping (call once per file)."""
        self.vad.reset()
        self.encoder.reset()
        self.online = OnlineClustering(**self._online_kwargs)
        # Embeddings awaiting a settled VAD decision, and the speech mask as
        # (first, last+1) frame ranges.
        self._embs: deque[torch.Tensor] = deque()
        self._next_emb = 0  # frame index of self._embs[0]
        self._speech: list[tuple[int, int]] = []
        self._sp_head = 0  # read cursor into _speech
        # Open (growing) output turn awaiting a label change / gap.
        self._open: list[float | int] | None = None  # [start, end, label]
        self._open_frame = -1  # last frame folded into _open
        self._speaker_of: dict[int, int] = {}  # cluster id -> output speaker number
        self._cluster_of: dict[int, int] = {}  # output speaker number -> cluster id

    def stream(self, audio, uri: str | None = None, sr: int | None = None) -> Iterator[str]:
        """Stream ``audio`` and yield an RTTM line per finalized speaker turn.

        Args:
            audio: 1-D mono waveform at 16 kHz (numpy array or torch tensor).
            uri: File id written in the RTTM lines (``<NA>`` when omitted).
            sr: Sample rate of ``audio``, checked against 16 kHz when given.

        Yields:
            RTTM lines (newline-terminated), in time order.
        """
        if sr is not None and sr != SAMPLE_RATE:
            raise ValueError(f"expected sr={SAMPLE_RATE}, got {sr}")
        uri = uri or "<NA>"
        if " " in uri:
            raise ValueError(f"RTTM does not allow file URIs containing spaces (got: {uri!r})")
        self.reset()
        wav = self._as_tensor(audio)
        for start in range(0, wav.shape[0], self.input_chunk):
            yield from self.to_rttm(self.push(wav[start : start + self.input_chunk]), uri)
        yield from self.to_rttm(self.finish(), uri)

    def __call__(self, audio, uri: str | None = None, sr: int | None = None) -> str:
        """Run the stream to completion and return the whole RTTM text."""
        return "".join(self.stream(audio, uri, sr))

    def push(self, chunk) -> list[tuple[float, float, int]]:
        """Feed the next chunk of a live stream: 16 kHz mono audio of any length.

        Call :meth:`reset` before the first chunk and :meth:`finish` after the
        last one.

        Returns:
            The ``(start_sec, end_sec, speaker)`` turns finalized by this chunk.
        """
        return self._process(self._as_tensor(chunk))

    def finish(self) -> list[tuple[float, float, int]]:
        """End the live stream and return its remaining turns."""
        return self._finalize()

    def _process(self, chunk: torch.Tensor) -> list[tuple[float, float, int]]:
        """Run one input chunk: the VAD masks the frames the encoder produces from it."""
        finalized: list[tuple[float, float, int]] = []
        if not chunk.numel():
            return finalized
        self._record_speech(self.vad(chunk))
        self._embs.extend(self.encoder.push(chunk))
        self._drain(self.vad.decided_upto(), finalized)
        return finalized

    def _finalize(self) -> list[tuple[float, float, int]]:
        """Settle the VAD, drain the encoder and flush the clusterer."""
        finalized: list[tuple[float, float, int]] = []
        self._record_speech(self.vad.flush())
        self._embs.extend(self.encoder.flush())
        self._drain(float("inf"), finalized)
        for frame, label in self.online.finalize():
            if not self.post_process:
                self._emit_frame(frame, label, finalized)
        if self.post_process:
            # offline merging relabels frames that were already clustered
            self.online.merge_subclusters()
            for frame, label in self.online.labels():
                self._emit_frame(frame, label, finalized)
        if self._open is not None:
            finalized.append((self._open[0], self._open[1], self._open[2]))
            self._open = None
        return finalized

    def _record_speech(self, intervals: list[tuple[int, int]]) -> None:
        """Convert confirmed speech samples to frame ranges of the encoder grid."""
        for start, end in intervals:
            first = round(start / self.frame_samples)
            last = round(end / self.frame_samples)
            if last > first:
                self._speech.append((first, last))

    def _drain(self, decided: float, finalized: list) -> None:
        """Cluster every buffered frame whose VAD decision can no longer change."""
        while self._embs and (self._next_emb + 1) * self.frame_samples <= decided:
            emb = self._embs.popleft().detach().cpu().numpy()
            frame, self._next_emb = self._next_emb, self._next_emb + 1
            for out_frame, label in self.online.fit(emb, self._is_speech(frame)):
                if not self.post_process:
                    self._emit_frame(out_frame, label, finalized)

    def _is_speech(self, frame: int) -> bool:
        """Speech mask of ``frame``; frames are queried in increasing order."""
        while self._sp_head < len(self._speech) and self._speech[self._sp_head][1] <= frame:
            self._sp_head += 1
        return self._sp_head < len(self._speech) and self._speech[self._sp_head][0] <= frame

    def similarity(self, frame: int, speaker: int) -> float:
        """Cosine similarity of an emitted ``frame`` to the current centroid of ``speaker``."""
        return self.online.similarity(frame, self._cluster_of[speaker])

    def _emit_frame(self, frame: int, cluster: int | None, finalized: list) -> None:
        """Append one labeled frame to the open turn, merging contiguous runs."""
        label = None
        if cluster is not None:
            label = self._speaker_of.get(cluster)
            if label is None:
                label = self._speaker_of[cluster] = len(self._speaker_of) + 1
                self._cluster_of[label] = cluster
        if self.on_frame is not None:
            self.on_frame(frame, label)
        if label is None:  # silence closes the current turn
            if self._open is not None:
                finalized.append((self._open[0], self._open[1], self._open[2]))
                self._open = None
            return
        start = frame * self.frame_shift
        end = start + self.frame_shift
        if self._open is not None and self._open[2] == label and frame == self._open_frame + 1:
            self._open[1] = end
        else:
            if self._open is not None:
                finalized.append((self._open[0], self._open[1], self._open[2]))
            self._open = [start, end, label]
        self._open_frame = frame

    @staticmethod
    def to_rttm(turns: list[tuple[float, float, int]], uri: str) -> Iterator[str]:
        """RTTM lines of ``(start_sec, end_sec, speaker)`` turns."""
        for start, end, label in turns:
            if end > start:
                yield (
                    f"SPEAKER {uri} 1 {start:.3f} {end - start:.3f} "
                    f"<NA> <NA> spk{int(label)} <NA> <NA>\n"
                )

    @staticmethod
    def _as_tensor(audio) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        return audio.reshape(-1).float()
