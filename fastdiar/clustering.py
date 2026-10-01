import heapq
from collections.abc import Iterator

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class _RunningMedian:
    """Exact per-dimension median of a growing set of vectors.

    Every dimension keeps its lower half in a max-heap and its upper half in a
    min-heap, so adding a vector costs O(d log n) instead of a new median over
    all n vectors (O(d n), which makes a whole stream quadratic). :meth:`value`
    equals ``np.median(vectors, axis=0)`` exactly.
    """

    def __init__(self, dim: int, dtype) -> None:
        self.dtype = dtype
        self._lower = [[] for _ in range(dim)]  # negated values: max-heaps
        self._upper = [[] for _ in range(dim)]
        self.count = 0

    def add(self, vec: np.ndarray) -> None:
        for x, lower, upper in zip(vec.tolist(), self._lower, self._upper, strict=True):
            if not lower or x <= -lower[0]:
                heapq.heappush(lower, -x)
            else:
                heapq.heappush(upper, x)
            if len(lower) > len(upper) + 1:
                heapq.heappush(upper, -heapq.heappop(lower))
            elif len(upper) > len(lower):
                heapq.heappush(lower, -heapq.heappop(upper))
        self.count += 1

    def value(self) -> np.ndarray:
        lower = np.array([-heap[0] for heap in self._lower], dtype=self.dtype)
        if self.count % 2:
            return lower
        upper = np.array([heap[0] for heap in self._upper], dtype=self.dtype)
        return np.mean(np.stack([lower, upper]), axis=0)  # np.median's arithmetic


class OnlineClustering:
    """Confidence-gated online clustering of per-frame speaker embeddings.

    A speech frame is confident once its look-back similarity ``e_t . e_{t-delay_frames}``
    reaches ``confidence``; confident frames become cluster anchors and label
    the pending frames that are leaving the window. The output lags the input
    by a *fixed* ``max_delay_sec``: frame ``t`` is emitted when frame
    ``t + max_delay_frames`` arrives, no earlier and no later. The frames in
    between are the self-correction window -- while they wait they can still
    inherit the label of a later confident frame or be relabeled by an online
    cluster merge, and only the label they hold when they leave the window is
    emitted.

    A frame that reaches the end of the window still unassigned has run out of
    context, so a decision is forced for it: non-speech (per the VAD label) is
    emitted as silence, and speech goes to the closest cluster by cosine
    similarity to the cluster medians, with no threshold to pass. Forced frames
    stay out of ``medians``, so they never become cluster anchors and a wrong
    guess does not drag a centroid along.
    """

    def __init__(
        self,
        delay_frames: int = 10,
        clust_th: float = 0.4,
        merge_th: float = 0.8,
        confidence: float = 0.8,
        sec_per_frame: float = 0.08,
        min_speech_sec: float = 2.0,
        post_process: bool = False,
        online_merge: bool = True,
        max_delay_sec: float = 0.96,
    ):
        self.delay_frames = delay_frames
        self.clust_th = clust_th
        self.merge_th = merge_th
        self.confidence = confidence
        self.sec_per_frame = sec_per_frame
        self.min_speech_sec = min_speech_sec
        self.post_process = post_process
        self.online_merge = online_merge
        # fixed output lag, in frames
        self.max_delay_frames = round(max_delay_sec / sec_per_frame)

        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.vad_labels = []
        self.clusters = {}
        self.speech_clusters = {}
        self.frame_labels = {}  # frame id -> cluster id, None for silence
        self.medians = []  # per cluster: median frame ids, None once merged away
        self.unprocessed = []
        self.next_frame = 0  # next frame id to emit
        self._buffer = None  # growing backing store of `embeddings`
        self._median_cache = {}  # cluster id -> [running median of its anchors, unit median]

    def process_file(self, embs, vad_labels):
        starts = np.arange(0, len(embs) * self.sec_per_frame, self.sec_per_frame)
        ends = starts + self.sec_per_frame
        for emb, vad_label in zip(embs, vad_labels, strict=True):
            for _ in self.fit(emb, vad_label):
                pass
        for _ in self.finalize():
            pass

        if not self.speech_clusters:
            return []

        # merge subclusters
        if self.post_process and len(self.speech_clusters) > 1:
            self.merge_subclusters()

        blocks = []
        # clusters to time segments
        for lbl, indices in self.speech_clusters.items():
            idx = np.array(indices)
            if len(idx) == 0:
                continue

            speech_starts = starts[idx]
            speech_ends = ends[idx]
            speech_duration = sum(speech_ends - speech_starts)

            if self.post_process and speech_duration < self.min_speech_sec:
                continue

            blocks.append(
                np.stack([speech_starts, speech_ends, np.full(len(speech_starts), lbl)], axis=1)
            )

        return blocks

    def merge_subclusters(self):
        """Offline pass: merge similar speech clusters and relabel their frames."""
        groups = self._similar_speech_groups()
        if not groups:
            return

        merged_speech_clusters = {}
        for lbl, group in groups.items():
            merged_speech_clusters[int(lbl)] = [
                idx for cur_lbl in group for idx in self.speech_clusters[cur_lbl]
            ]
        self.speech_clusters = merged_speech_clusters

        for lbl, indices in merged_speech_clusters.items():
            for idx in indices:
                self.frame_labels[idx] = lbl

    def labels(self) -> Iterator[tuple[int, int | None]]:
        """Current label of every already assigned frame, in frame order."""
        for idx in range(len(self.embeddings)):
            if idx in self.frame_labels:
                yield idx, self.frame_labels[idx]

    def similarity(self, idx: int, cluster: int) -> float:
        """Cosine similarity of frame ``idx`` to the median of ``cluster`` (NaN once merged)."""
        ids, medians = self._cluster_medians()
        if cluster not in ids:
            return float("nan")
        return float(self.embeddings[idx] @ medians[ids.index(cluster)])

    def fit(self, emb, vad_label) -> Iterator[tuple[int, int | None]]:
        """Consume one frame, yield ``(frame_id, cluster_id)`` for finalized frames.

        Output lags the input by exactly ``max_delay_frames``: every frame is
        held for the full window, which is what makes the self-corrections
        inside it possible. Silence frames are yielded as ``(frame_id, None)``.
        """
        i = len(self.embeddings)
        self._append(emb)
        self.vad_labels.append(vad_label)

        self._assign(i)
        self._force_assign(i)
        yield from self._emit(i - self.max_delay_frames)

    def finalize(self) -> Iterator[tuple[int, int | None]]:
        """Flush the frames still pending at end-of-stream.

        Remaining speech frames go to their most similar cluster (silence is
        dropped); if no cluster exists yet the frames are ignored.
        """
        pending = self.unprocessed
        self.unprocessed = []

        ids, medians = self._cluster_medians()
        if pending and ids:
            for idx in pending:
                if not self.vad_labels[idx]:
                    self.frame_labels[idx] = None
                    continue
                c = ids[int(np.argmax(self.embeddings[idx] @ medians.T))]
                self.clusters[c].append(idx)
                self.speech_clusters.setdefault(c, []).append(idx)
                self.frame_labels[idx] = c

        yield from self._emit(len(self.embeddings) - 1)

    def _assign(self, i):
        # warm-up: no look-back available yet
        if i < self.delay_frames:
            self.unprocessed.append(i)
            return

        # silence is never an anchor: it would open, or pull, a cluster with no speaker
        emb = self.embeddings[i]
        if not self.vad_labels[i] or emb @ self.embeddings[i - self.delay_frames] < self.confidence:
            self.unprocessed.append(i)
            return

        # compare with current clusters
        c = len(self.medians)  # default: open a new cluster
        ids, medians = self._cluster_medians()
        if ids:
            sims = emb @ medians.T
            best = int(np.argmax(sims))
            if sims[best] >= self.clust_th:
                c = ids[best]

        # delayed assignment: frames inside the window always stay pending
        cutoff = i - self.max_delay_frames
        ready = [idx for idx in self.unprocessed if idx <= cutoff]
        self.unprocessed = [idx for idx in self.unprocessed if idx > cutoff]
        self.unprocessed.append(i)

        if c == len(self.medians):
            # new cluster
            self.clusters[c] = ready
            self.medians.append([i])
        else:
            # existing cluster
            self.clusters[c].extend(ready)
            self.medians[c].append(i)

        # vad filtering
        speech_indices = []
        for idx in ready:
            if self.vad_labels[idx]:
                speech_indices.append(idx)
                self.frame_labels[idx] = c
            else:
                self.frame_labels[idx] = None

        new_speech_cluster = False
        if len(speech_indices) > 0:
            if c in self.speech_clusters:
                self.speech_clusters[c].extend(speech_indices)
            else:
                self.speech_clusters[c] = speech_indices
                new_speech_cluster = True

        # merge speech clusters
        if self.online_merge and not self.post_process and new_speech_cluster:
            self._merge_speech_clusters()

    def _force_assign(self, i):
        """Empty the self-correction window: decide the frames leaving it.

        Whatever is still pending ``max_delay_frames`` back never passed the
        confidence gate and has no context left to wait for, so a label is
        forced (see the class docstring). Forced frames join ``clusters`` and
        ``speech_clusters`` but never ``medians``, so they do not become anchors
        of the cluster they were guessed into.
        """
        cutoff = i - self.max_delay_frames
        if cutoff < 0 or not self.unprocessed:
            return
        ready = [idx for idx in self.unprocessed if idx <= cutoff]
        if not ready:
            return
        self.unprocessed = [idx for idx in self.unprocessed if idx > cutoff]

        new_speech_cluster = False
        for idx in ready:
            # vad filtering: no speech, no speaker
            if not self.vad_labels[idx]:
                self.frame_labels[idx] = None
                continue

            c = self._nearest_cluster(idx)
            self.clusters[c].append(idx)
            if c in self.speech_clusters:
                self.speech_clusters[c].append(idx)
            else:
                self.speech_clusters[c] = [idx]
                new_speech_cluster = True
            self.frame_labels[idx] = c

        # merge speech clusters
        if self.online_merge and not self.post_process and new_speech_cluster:
            self._merge_speech_clusters()

    def _nearest_cluster(self, idx):
        """Closest cluster to frame ``idx``, opening one when there is none."""
        # unconditionally: no clust_th to pass
        ids, medians = self._cluster_medians()
        if ids:
            return ids[int(np.argmax(self.embeddings[idx] @ medians.T))]

        c = len(self.medians)
        self.clusters[c] = []
        self.medians.append([idx])
        return c

    def _merge_speech_clusters(self):
        for group in self._similar_speech_groups().values():
            if len(group) == 1:
                continue
            new_lbl = min(group)
            for lbl in group:
                if lbl == new_lbl:
                    continue
                # relabel the moved frames: already emitted ones keep the old id,
                # pending ones are emitted with the merged id
                for idx in self.speech_clusters[lbl]:
                    self.frame_labels[idx] = new_lbl
                self.speech_clusters[new_lbl].extend(self.speech_clusters[lbl])
                self.speech_clusters[lbl] = []

                # fuse the cluster state, otherwise the merged id keeps attracting
                # frames and the merge has no effect downstream
                self.clusters[new_lbl].extend(self.clusters[lbl])
                self.clusters[lbl] = []
                self.medians[new_lbl].extend(self.medians[lbl])
                self.medians[lbl] = None
                self._median_cache.pop(lbl, None)

    def _similar_speech_groups(self):
        """Group non-empty speech clusters whose medians exceed ``merge_th``.

        Returns ``{component id: [cluster ids]}``, or ``{}`` when no pair is
        similar enough to merge.
        """
        clust_ids, speech_medians = [], []
        for lbl, indices in self.speech_clusters.items():
            if len(indices) == 0:
                continue
            clust_ids.append(lbl)
            speech_medians.append(np.median(self.embeddings[indices], axis=0))
        if len(clust_ids) < 2:
            return {}

        speech_medians = np.asarray(speech_medians)
        speech_medians /= np.linalg.norm(speech_medians, axis=1, keepdims=True)
        sims = speech_medians @ speech_medians.T

        rows, cols = np.where(np.tril(sims, k=-1) > self.merge_th)
        if len(rows) == 0:
            return {}

        n = sims.shape[0]
        adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _, labels = connected_components(adj, directed=False)

        groups = {}
        for cur_lbl, new_lbl in zip(clust_ids, labels, strict=True):
            groups.setdefault(new_lbl, []).append(cur_lbl)
        return groups

    def _append(self, emb):
        emb = np.asarray(emb)
        n = len(self.embeddings)
        if self._buffer is None:
            self._buffer = np.empty((16, emb.shape[0]), dtype=emb.dtype)
        elif n == len(self._buffer):
            grown = np.empty((2 * n, self._buffer.shape[1]), dtype=self._buffer.dtype)
            grown[:n] = self._buffer
            self._buffer = grown
        self._buffer[n] = emb
        self.embeddings = self._buffer[: n + 1]

    def _cluster_medians(self):
        """``(cluster ids, unit medians)`` of the clusters still alive."""
        ids, medians = [], []
        for c, idx in enumerate(self.medians):
            if idx is None:  # merged into another cluster
                continue
            cached = self._median_cache.get(c)
            if cached is None:
                running = _RunningMedian(self.embeddings.shape[1], self.embeddings.dtype)
                cached = self._median_cache[c] = [running, None]
            running = cached[0]
            if running.count != len(idx):
                # anchor lists only grow at the end (a merge extends them too)
                for i in idx[running.count :]:
                    running.add(self.embeddings[i])
                median = running.value()
                cached[1] = median / np.linalg.norm(median)
            ids.append(c)
            medians.append(cached[1])
        return ids, np.stack(medians) if medians else None

    def _emit(self, last_frame) -> Iterator[tuple[int, int | None]]:
        while self.next_frame <= last_frame and self.next_frame in self.frame_labels:
            yield self.next_frame, self.frame_labels[self.next_frame]
            self.next_frame += 1
