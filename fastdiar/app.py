# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gradio demo of the streaming speaker diarizer.

Usage:
    python -m fastdiar.app                 # large model, http://127.0.0.1:7860
    python -m fastdiar.app -m small --share

The audio is an uploaded file or the microphone. Either way it is diarized as a
stream: an upload is played in the browser while it is fed chunk by chunk at the
same real-time pace (or as fast as possible, without playback), and the
microphone records from the moment "Start" is pressed. The waveform of
the last 20 s is colored by speaker as the frame labels are finalized, about
one second behind the audio (the clustering delay plus the VAD decision), with
the assignment confidence of the latest frame on the right. The whole stream is
kept: the mouse wheel, dragging the waveform or the bar below it scroll back to
past frames. "Stop" ends the stream and offers the RTTM for download.
"""

import argparse
import math
import secrets
import tempfile
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchaudio

from fastdiar.cli import add_device_args, add_model_args, load_audio, load_model
from fastdiar.diarizer import SAMPLE_RATE, StreamingDiarizer

SHIFT_SEC = 0.32  # streaming step, also on a GPU: a live stream's output cadence
WINDOW_SEC = 20.0
COL_SAMPLES = 320  # 20 ms per waveform column
N_COLS = round(WINDOW_SEC * SAMPLE_RATE / COL_SAMPLES)
WAVE_HEIGHT = 160  # SVG units == px
TICK_SEC = 5
PALETTE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#393b79", "#ad494a", "#637939",
)  # fmt: skip

# Browser side of Start / Stop. An uploaded file is played from the start as Start is
# pressed, while the server streams it at the same pace. The microphone component is
# hidden and driven by the buttons: its record button is pressed once the session has
# started, and its stop button when Stop is pressed.
_UPLOAD_AUDIO_JS = """const uploadAudio = () => {
  for (const el of document.querySelectorAll("#fd-upload *")) {
    const audio = el.shadowRoot?.querySelector("audio");  // the waveform player's element
    if (audio) return audio;
  }
  return document.querySelector("#fd-upload audio");
};"""
START_JS = f"""(source, path, realtime) => {{
  {_UPLOAD_AUDIO_JS}
  const audio = uploadAudio();
  if (audio) {{
    audio.pause();
    if (source === "Upload" && path && realtime) {{
      audio.currentTime = 0;
      audio.playbackRate = 1;
      audio.play();
    }}
  }}
  return [source, path, realtime];
}}"""
MIC_START_JS = """(source) => {
  if (source === "Microphone") document.querySelector("#fd-mic .record-button")?.click();
}"""
STOP_JS = f"""() => {{
  {_UPLOAD_AUDIO_JS}
  uploadAudio()?.pause();
  document.querySelector("#fd-mic .stop-button")?.click();
}}"""

# Scrolling back through the stream: the wheel or dragging the waveform pans it, and
# the bar below it works as a scrollbar. The session keeps the whole history and the
# window to show (see `Session.view_end`); the browser asks for another window with
# `scroll(token, end)` and shows the result. Every live update re-renders the view, so
# the listeners sit on the component root, which is kept.
SCROLL_JS = f"""
const N = {N_COLS};
let pending, inflight, busy = false, drag = null;
const view = () => element.querySelector(".fd-view");
const cols = () => Number(view()?.dataset.cols || 0);
const shown = (end) => (end === null ? cols() : end);
const current = () =>
  pending !== undefined ? shown(pending) : busy ? shown(inflight) : Number(view().dataset.end);
function send() {{
  if (busy || pending === undefined) return;
  inflight = pending;
  pending = undefined;
  busy = true;
  server.scroll(view().dataset.token, inflight)
    .then((html) => {{ if (html) props.value = html; }})
    .finally(() => {{ busy = false; send(); }});
}}
function scrollTo(end) {{  // end column of the window; the last one follows the stream
  const n = cols();
  if (n <= N) return;
  end = Math.round(Math.min(Math.max(end, N), n));
  pending = end >= n ? null : end;
  send();
}}
element.addEventListener("wheel", (e) => {{
  const target = e.target.closest(".fd-wave, .fd-scrollbar");
  if (!target || cols() <= N) return;
  e.preventDefault();
  const d = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
  scrollTo(current() + (d * N) / element.querySelector(".fd-wave").clientWidth);
}}, {{ passive: false }});
element.addEventListener("pointerdown", (e) => {{
  const bar = e.target.closest(".fd-scrollbar");
  const wave = e.target.closest(".fd-wave svg");
  if ((!bar && !wave) || cols() <= N) return;
  let end = current(), scale;
  if (bar) {{
    const rect = bar.getBoundingClientRect();
    scale = cols() / rect.width;
    if (!e.target.closest(".fd-thumb")) {{  // jump: center the window on the pointer
      end = (e.clientX - rect.left) * scale + N / 2;
      scrollTo(end);
    }}
  }} else {{
    scale = -N / wave.getBoundingClientRect().width;  // drag the waveform itself
  }}
  drag = {{ x: e.clientX, end, scale }};
  element.setPointerCapture(e.pointerId);
  e.preventDefault();
}});
element.addEventListener("pointermove", (e) => {{
  if (drag) scrollTo(drag.end + (e.clientX - drag.x) * drag.scale);
}});
element.addEventListener("pointerup", () => {{ drag = null; }});
element.addEventListener("pointercancel", () => {{ drag = null; }});
element.addEventListener("click", (e) => {{
  if (e.target.closest(".fd-live-btn")) scrollTo(Infinity);
}});
"""

CSS = """
#fd-mic { display: none !important; }
.fd-view { font-family: var(--font); color: var(--body-text-color); }
.fd-top { display: flex; align-items: center; gap: 10px; min-height: 26px; margin-bottom: 6px; }
.fd-status { font-size: 13px; color: var(--body-text-color-subdued); }
.fd-live-btn { font-size: 12px; padding: 2px 10px; border-radius: 999px; cursor: pointer;
  border: 1px solid var(--border-color-primary); background: var(--background-fill-primary);
  color: var(--body-text-color); }
.fd-speakers { display: flex; flex-wrap: wrap; gap: 6px; min-height: 30px; margin-bottom: 8px; }
.fd-chip { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px;
  border-radius: 999px; border: 1px solid var(--border-color-primary); font-size: 13px; }
.fd-chip i { width: 10px; height: 10px; border-radius: 50%; background: var(--c); }
.fd-chip.active { border-color: var(--c); box-shadow: 0 0 0 1px var(--c); font-weight: 600; }
.fd-hint { font-size: 13px; color: var(--body-text-color-subdued); align-self: center; }
.fd-row { display: flex; gap: 10px; align-items: stretch; }
.fd-wave { flex: 1; min-width: 0; position: relative; }
.fd-wave svg { display: block; width: 100%; height: 160px; border-radius: 6px;
  background: var(--background-fill-secondary); touch-action: pan-y; }
.fd-wave.scrollable svg { cursor: grab; }
.fd-scrollbar { position: relative; height: 8px; margin-top: 4px; border-radius: 4px;
  background: var(--background-fill-secondary); cursor: pointer; touch-action: none; }
.fd-thumb { position: absolute; top: 0; bottom: 0; min-width: 12px; border-radius: 4px;
  background: var(--body-text-color-subdued); opacity: 0.55; }
.fd-pending { stroke: var(--body-text-color); opacity: 0.45; }
.fd-silence { stroke: var(--body-text-color); opacity: 0.18; }
.fd-axis { position: relative; height: 18px; font-size: 11px;
  color: var(--body-text-color-subdued); }
.fd-axis span { position: absolute; transform: translateX(-50%); top: 2px; }
.fd-conf { width: 46px; display: flex; flex-direction: column; align-items: center; gap: 4px; }
.fd-conf-track { position: relative; width: 18px; height: 160px; border-radius: 6px;
  background: var(--background-fill-secondary); overflow: hidden; }
.fd-conf-fill { position: absolute; bottom: 0; left: 0; right: 0; background: var(--c); }
.fd-conf-value { font-size: 12px; font-variant-numeric: tabular-nums; }
.fd-conf-label { font-size: 11px; color: var(--body-text-color-subdued); }
"""


class StreamResampler:
    """Chunk-wise resampling to 16 kHz, equal to resampling the whole stream at once.

    Every call resamples the new input together with enough context on both
    sides for the sinc filter, and only outputs the samples whose filter support
    lies inside it; the right context is held back until the next call.
    """

    def __init__(self, orig_sr: int) -> None:
        self.orig_sr = orig_sr
        g = math.gcd(orig_sr, SAMPLE_RATE)
        self.up, self.down = orig_sr // g, SAMPLE_RATE // g  # input / output block
        # Half-width of torchaudio's default filter (lowpass_filter_width=6, rolloff=0.99),
        # in input samples, rounded up to whole blocks.
        width = math.ceil(6 * self.up / (min(self.up, self.down) * 0.99)) + 1
        self.ctx = math.ceil(width / self.up) * self.up
        self._buf = torch.zeros(self.ctx)  # the stream starts with zero padding

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.orig_sr == SAMPLE_RATE:
            return x
        self._buf = torch.cat([self._buf, x])
        usable = (self._buf.shape[0] - 2 * self.ctx) // self.up * self.up
        if usable <= 0:
            return torch.zeros(0)
        y = torchaudio.functional.resample(
            self._buf[: usable + 2 * self.ctx], self.orig_sr, SAMPLE_RATE
        )
        start = self.ctx // self.up * self.down
        out = y[start : start + usable // self.up * self.down]
        self._buf = self._buf[usable:]
        return out

    def flush(self) -> torch.Tensor:
        """The held-back tail, with the end of the stream zero-padded."""
        if self.orig_sr == SAMPLE_RATE:
            return torch.zeros(0)
        tail = self._buf.shape[0] - self.ctx
        pad = self.ctx + (-tail) % self.up
        out = self(torch.zeros(pad))
        return out[: math.ceil(tail * self.down / self.up)]


class Session:
    """The diarization stream of one browser session and what is displayed of it."""

    def __init__(self, model) -> None:
        self.lock = threading.Lock()
        self.token = secrets.token_hex(16)  # names the session in `scroll` requests
        self.diarizer = StreamingDiarizer(model, shift_sec=SHIFT_SEC, on_frame=self._on_frame)
        self.frame_cols = self.diarizer.frame_samples // COL_SAMPLES
        self.out_dir = Path(tempfile.mkdtemp(prefix="fastdiar_"))
        self.generation = 0
        self.rttm_path: Path | None = None
        self._clear()
        # Warm up this session's VAD and encoder state before the first stream.
        self.diarizer(np.random.default_rng(0).normal(0, 0.1, SAMPLE_RATE).astype(np.float32))
        self._clear()
        self.status = "Ready"

    def _clear(self) -> None:
        self.running = False
        self.source = None
        self.uri = "<NA>"
        self.resampler: StreamResampler | None = None
        self.samples = 0  # 16 kHz samples received
        self.view_end: int | None = None  # end column of the shown window; None: follow
        self.peaks: list[float] = []  # peak amplitude of every 20 ms column
        self._col_tail = np.zeros(0, dtype=np.float32)
        self.labels: list[int | None] = []  # finalized label of every frame
        self.colors: dict[int, str] = {}  # speaker -> color, in order of appearance
        self.confidence: float | None = None
        self.turns: list[tuple[float, float, int]] = []

    def start(self, source: str, uri: str) -> int:
        """Reset for a new stream; returns its generation, which ``stop`` invalidates."""
        self._clear()
        self.diarizer.reset()
        self.running, self.source, self.uri = True, source, uri
        self.rttm_path = None
        self.status = "Listening to the microphone" if source == "mic" else f"Streaming {uri}"
        self.generation += 1
        return self.generation

    def feed(self, wav: np.ndarray) -> None:
        """Push 16 kHz audio through the diarizer and into the waveform."""
        if not wav.size:
            return
        self.samples += wav.size
        cols = np.concatenate([self._col_tail, wav])
        n = cols.size // COL_SAMPLES * COL_SAMPLES
        self.peaks.extend(np.abs(cols[:n]).reshape(-1, COL_SAMPLES).max(axis=1).tolist())
        self._col_tail = cols[n:]
        self.turns.extend(self.diarizer.push(wav))

    def feed_mic(self, sr: int, data: np.ndarray) -> None:
        wav = data.astype(np.float32)
        if np.issubdtype(data.dtype, np.integer):
            wav /= np.iinfo(data.dtype).max + 1
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if self.resampler is None or self.resampler.orig_sr != sr:
            self.resampler = StreamResampler(sr)
        self.feed(self.resampler(torch.from_numpy(wav)).numpy())

    def stop(self) -> Path:
        """End the stream and write its RTTM."""
        if self.resampler is not None:
            self.feed(self.resampler.flush().numpy())
        self.turns.extend(self.diarizer.finish())
        self.running = False
        self.rttm_path = self.out_dir / f"{self.uri}.rttm"
        self.rttm_path.write_text("".join(self.diarizer.to_rttm(self.turns, self.uri)))
        self.status = "Stopped"
        return self.rttm_path

    def render(self) -> str:
        return render(self)

    def _on_frame(self, frame: int, label: int | None) -> None:
        self.labels.append(label)
        if label is None:
            self.confidence = None
            return
        self.colors.setdefault(label, PALETTE[len(self.colors) % len(PALETTE)])
        self.confidence = self.diarizer.similarity(frame, label)


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def render(s: Session) -> str:
    """HTML of the speaker chips, the rolling waveform and the confidence bar."""
    t_now = s.samples / SAMPLE_RATE
    n_cols = len(s.peaks)
    live = s.view_end is None
    col_end = n_cols if live else min(max(s.view_end, N_COLS), n_cols)
    col0 = max(0, col_end - N_COLS)
    t0 = col0 * COL_SAMPLES / SAMPLE_RATE
    peaks = s.peaks[col0:col_end]
    active = s.labels[-1] if s.labels and s.running else None

    chips = (
        "".join(
            f'<span class="fd-chip{" active" if spk == active else ""}" style="--c:{color}">'
            f"<i></i>spk{spk}</span>"
            for spk, color in s.colors.items()
        )
        or '<span class="fd-hint">Speakers appear here as they are detected</span>'
    )

    # One vertical stroke per column, grouped into a path per color.
    scale = max(max(peaks, default=0.0), 0.02)
    half = WAVE_HEIGHT / 2
    paths: dict[str, list[str]] = defaultdict(list)
    n_labeled = len(s.labels)
    for i, peak in enumerate(peaks):
        frame = (col0 + i) // s.frame_cols
        if frame >= n_labeled:
            key = "pending"
        else:
            label = s.labels[frame]
            key = "silence" if label is None else s.colors[label]
        h = max(peak / scale, 0.01) * (half - 6)
        paths[key].append(f"M{i + 0.5} {half - h:.1f}V{half + h:.1f}")
    strokes = []
    for key, segs in paths.items():
        style = f'class="fd-{key}"' if key in ("pending", "silence") else f'stroke="{key}"'
        strokes.append(f'<path {style} stroke-width="0.8" d="{"".join(segs)}"/>')

    # Time grid, and the frontier of finalized labels.
    ticks = np.arange(math.ceil(t0 / TICK_SEC) * TICK_SEC, t0 + WINDOW_SEC + 1e-6, TICK_SEC)

    def to_x(t: float) -> float:
        return (t - t0) / WINDOW_SEC * N_COLS

    grid = "".join(
        f'<line x1="{to_x(t):.1f}" x2="{to_x(t):.1f}" y1="0" y2="{WAVE_HEIGHT}" '
        'stroke="var(--border-color-primary)" vector-effect="non-scaling-stroke"/>'
        for t in ticks
    )
    labels_html = "".join(
        f'<span style="left:{(t - t0) / WINDOW_SEC * 100:.2f}%">{_clock(t)}</span>'
        for t in ticks
        if 0 < t - t0 < WINDOW_SEC
    )
    frontier = ""
    if s.running and col0 <= n_labeled * s.frame_cols < col_end:
        x = to_x(n_labeled * s.frame_cols * COL_SAMPLES / SAMPLE_RATE)
        frontier = (
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{WAVE_HEIGHT}" '
            'stroke="var(--body-text-color)" stroke-dasharray="4 3" opacity="0.6" '
            'vector-effect="non-scaling-stroke"/>'
        )

    conf = s.confidence
    conf_color = s.colors.get(active, "var(--body-text-color-subdued)")
    fill = 0.0 if conf is None or math.isnan(conf) else min(max(conf, 0.0), 1.0)
    conf_text = "–" if conf is None or math.isnan(conf) else f"{conf:.2f}"

    n_spk = len(s.colors)
    status = s.status
    if s.samples:
        status += f" · {_clock(t_now)} · {n_spk} speaker{'s' if n_spk != 1 else ''}"
    live_btn = ""
    if not live:
        status += f" · showing {_clock(t0)}–{_clock(t0 + WINDOW_SEC)}"
        live_btn = '<button class="fd-live-btn">Back to live ⏭</button>'

    scrollable = n_cols > N_COLS
    scrollbar = ""
    if scrollable:
        scrollbar = (
            f'<div class="fd-scrollbar"><div class="fd-thumb" style="left:{col0 / n_cols:.2%};'
            f'width:{N_COLS / n_cols:.2%}"></div></div>'
        )
    return f"""
<div class="fd-view" data-token="{s.token}" data-cols="{n_cols}" data-end="{col_end}">
  <div class="fd-top"><div class="fd-status">{status}</div>{live_btn}</div>
  <div class="fd-speakers">{chips}</div>
  <div class="fd-row">
    <div class="fd-wave{" scrollable" if scrollable else ""}">
      <svg viewBox="0 0 {N_COLS} {WAVE_HEIGHT}" preserveAspectRatio="none">
        {grid}{"".join(strokes)}{frontier}
      </svg>
      <div class="fd-axis">{labels_html}</div>
      {scrollbar}
    </div>
    <div class="fd-conf" title="Cosine similarity of the latest frame to its speaker">
      <div class="fd-conf-track">
        <div class="fd-conf-fill" style="height:{fill * 100:.0f}%;--c:{conf_color}"></div>
      </div>
      <div class="fd-conf-value">{conf_text}</div>
      <div class="fd-conf-label">conf.</div>
    </div>
  </div>
</div>"""


def build_demo(model) -> gr.Blocks:
    sessions: dict[str, Session] = {}  # by gradio session hash
    by_token: dict[str, Session] = {}  # by Session.token, for `scroll`
    sessions_lock = threading.Lock()

    def get_session(request: gr.Request) -> Session:
        with sessions_lock:
            if request.session_hash not in sessions:
                s = sessions[request.session_hash] = Session(model)
                by_token[s.token] = s
            return sessions[request.session_hash]

    def on_load(request: gr.Request):
        return get_session(request).render()

    def on_unload(request: gr.Request):
        with sessions_lock:
            s = sessions.pop(request.session_hash, None)
            if s is not None:
                by_token.pop(s.token, None)

    def scroll(args: list) -> str | None:
        """``server.scroll(token, end)``: show the window ending at column ``end``.

        ``end`` is ``None`` to follow the stream. Gradio passes the JS arguments as a list.
        """
        token, end = args
        with sessions_lock:
            s = by_token.get(token)
        if s is None:
            return None
        with s.lock:
            s.view_end = None if end is None or end >= len(s.peaks) else int(end)
            return s.render()

    def on_source(source: str):
        upload = source == "Upload"
        return gr.update(visible=upload), gr.update(visible=upload)

    def on_start(source: str, path: str | None, realtime: bool, request: gr.Request):
        # The browser starts playing the file now (START_JS): pace the stream from here,
        # so the time spent loading it is caught up instead of delaying the labels.
        started = time.perf_counter()
        s = get_session(request)
        if source == "Microphone":
            with s.lock:
                s.start("mic", f"mic_{datetime.now().astimezone():%Y%m%d_%H%M%S}")
            yield s.render(), None  # the microphone starts recording only now (MIC_START_JS)
            return

        if not path:
            raise gr.Error("Upload an audio file first.")
        wav = load_audio(Path(path)).numpy()
        with s.lock:
            # RTTM fields are space-separated, so the file id cannot contain spaces.
            generation = s.start("upload", Path(path).stem.replace(" ", "_"))
        yield s.render(), None

        chunk = s.diarizer.input_chunk
        last_yield = 0.0
        for pos in range(0, wav.size, chunk):
            if realtime:  # this chunk is complete once the stream reaches its end
                time.sleep(max(0.0, started + (pos + chunk) / SAMPLE_RATE - time.perf_counter()))
            with s.lock:
                if s.generation != generation:
                    return  # restarted by another Start
                if not s.running:
                    break  # stopped by Stop, which wrote the RTTM
                s.feed(wav[pos : pos + chunk])
            now = time.perf_counter()
            if realtime or now - last_yield > 0.1:
                last_yield = now
                yield s.render(), gr.skip()
        else:
            with s.lock:
                if s.generation != generation:
                    return
                if s.running:
                    s.stop()
        # The final view, also after Stop: an update of this stream still in flight
        # could otherwise land after Stop's and overwrite it.
        with s.lock:
            view, rttm_path = s.render(), s.rttm_path
        yield view, str(rttm_path)

    def on_mic(chunk, request: gr.Request):
        s = get_session(request)
        if chunk is None:
            return gr.skip()
        with s.lock:
            if not s.running or s.source != "mic":
                return gr.skip()  # audio from before Start is not used
            s.feed_mic(*chunk)
            return s.render()

    def on_stop(request: gr.Request):
        s = get_session(request)
        with s.lock:
            if s.running:
                s.stop()
            rttm_path = s.rttm_path
            return s.render(), None if rttm_path is None else str(rttm_path)

    with gr.Blocks(title="fastdiar") as demo:
        gr.Markdown(
            "# Streaming speaker diarization\n"
            "Upload a file or use the microphone, then press **Start**: the file is played "
            "and diarized as a live stream. The waveform is "
            "colored by speaker as the labels are finalized, about one second behind the "
            "audio; scroll it (wheel or drag) to see the past. **Stop** ends the stream and "
            "makes the RTTM available."
        )
        source = gr.Radio(["Upload", "Microphone"], value="Upload", label="Audio source")
        upload = gr.Audio(
            sources=["upload"], type="filepath", label="Audio file", elem_id="fd-upload"
        )
        # Hidden: recording is started and stopped by the Start / Stop buttons.
        mic = gr.Audio(sources=["microphone"], type="numpy", streaming=True, elem_id="fd-mic")
        realtime = gr.Checkbox(True, label="Play the file and stream it at real-time pace")
        with gr.Row():
            start = gr.Button("Start", variant="primary")
            stop = gr.Button("Stop", variant="stop")
        view = gr.HTML(js_on_load=SCROLL_JS, server_functions=[scroll])
        rttm = gr.File(label="RTTM", interactive=False)

        source.change(on_source, source, [upload, realtime])
        start.click(
            on_start, [source, upload, realtime], [view, rttm], js=START_JS, concurrency_limit=None
        ).then(None, source, None, js=MIC_START_JS)
        mic.stream(
            on_mic,
            mic,
            view,
            stream_every=SHIFT_SEC,
            time_limit=None,
            concurrency_limit=None,
            show_progress="hidden",
        )
        stop.click(on_stop, None, [view, rttm], js=STOP_JS, concurrency_limit=None)
        demo.load(on_load, None, view)
        demo.unload(on_unload)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser)
    add_device_args(parser)
    parser.add_argument("--host", default="127.0.0.1", help="server address")
    parser.add_argument("--port", type=int, default=7860, help="server port")
    parser.add_argument("--share", action="store_true", help="create a public gradio link")
    args = parser.parse_args()

    model = load_model(args)
    # Warm up the model before serving, so the first stream runs at full speed.
    warmup = np.random.default_rng(0).normal(0, 0.1, 3 * SAMPLE_RATE).astype(np.float32)
    StreamingDiarizer(model, shift_sec=SHIFT_SEC)(warmup)
    build_demo(model).queue().launch(
        server_name=args.host, server_port=args.port, share=args.share, css=CSS
    )


if __name__ == "__main__":
    main()
