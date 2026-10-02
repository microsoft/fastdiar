# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Evaluation dataset loaders.

Every loader takes the dataset's audio directory and labels path and returns a
list of ``(audio_path, uri, reference)`` items; :data:`DATASETS` maps the
dataset names to them. See ``README.md`` for how to obtain each dataset.
"""

import json
from collections.abc import Callable, Iterable
from pathlib import Path

from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm

Item = tuple[Path, str, Annotation]

# Reserved RAMC "speaker" id tagging non-speaker events ([LAUGHTER], [MUSIC],
# [*] unintelligible, [+] overlap markers); dropped from the reference.
RAMC_NON_SPEAKER = "G00000000"


def load_reference(rttm_path: Path, uri: str) -> Annotation | None:
    """Load the reference :class:`Annotation` for ``uri`` from an RTTM file.

    Falls back to the file's single annotation when its uri column does not
    match the audio filename stem.
    """
    rttm = load_rttm(str(rttm_path))
    if not rttm:
        return None
    if uri in rttm:
        return rttm[uri]
    return next(iter(rttm.values()))


def annotation_from_spans(uri: str, spans: Iterable[tuple[float, float, str]]) -> Annotation:
    """Build a reference from ``(start, end, speaker)`` spans, merging each speaker's overlaps."""
    reference = Annotation(uri=uri)
    for start, end, speaker in spans:
        if end > start:
            reference[Segment(start, end), speaker] = speaker
    return reference.support()


def collect_items(
    uris: Iterable[str],
    audio_path: Callable[[str], Path],
    reference: Callable[[str], Annotation | None],
) -> list[Item]:
    """Pair every uri's audio with its reference, skipping files missing either."""
    items: list[Item] = []
    for uri in uris:
        path = audio_path(uri)
        if not path.is_file():
            print(f"[skip] missing audio for {uri}: {path}")
            continue
        ref = reference(uri)
        if not ref:
            print(f"[skip] empty reference for {uri}")
            continue
        ref.uri = uri
        items.append((path, uri, ref))
    if not items:
        raise FileNotFoundError("no files with both audio and a reference")
    return items


def rttm_dir_items(audio_dir: Path, rttm_dir: Path, audio_suffix: str) -> list[Item]:
    """Items of per-file ``<rttm_dir>/<uri>.rttm`` labels and ``<audio_dir>/<uri><suffix>`` audio.

    The RTTM files define the evaluated set; shared by AMI, DIHARD and VoxConverse.
    """
    uris = sorted(p.stem for p in rttm_dir.glob("*.rttm"))
    if not uris:
        raise FileNotFoundError(f"no .rttm files in {rttm_dir}")
    return collect_items(
        uris,
        lambda uri: audio_dir / f"{uri}{audio_suffix}",
        lambda uri: load_reference(rttm_dir / f"{uri}.rttm", uri),
    )


def load_ami(audio_dir: Path, labels: Path) -> list[Item]:
    """AMI meeting corpus, IHM-MIX condition (headset mixdown) from ami-corpus-mirror.

    Audio: ``<audio_dir>/<meeting>.Mix-Headset.wav`` -- the ``sdm/`` directory of
    the ami-corpus-mirror, which holds the 16 meetings of the official evaluation
    split. Labels: ``<labels>/<meeting>.rttm`` -- ``only_words/rttms/test`` of
    pyannote/AMI-diarization-setup (manual annotations, pauses between words are
    not speech).
    """
    items = rttm_dir_items(audio_dir, labels, ".Mix-Headset.wav")
    return [(path, uri, reference.support()) for path, uri, reference in items]


def load_dihard(audio_dir: Path, labels: Path) -> list[Item]:
    """DIHARD III (eval or dev).

    Audio: ``<audio_dir>/<uri>.flac`` (``data/flac``). Labels: ``<labels>/<uri>.rttm``
    (``data/rttm``). Scoring the whole file with collar 0 and overlap included
    matches the DIHARD III "full" protocol.
    """
    return rttm_dir_items(audio_dir, labels, ".flac")


def load_msdwild(audio_dir: Path, labels: Path) -> list[Item]:
    """MSDWild.

    Audio: ``<audio_dir>/<uri>.wav``. Labels: one combined RTTM per evaluation
    protocol (``few.val.rttm`` or ``many.val.rttm``); only the clips it references
    are evaluated.
    """
    references = load_rttm(str(labels))
    if not references:
        raise ValueError(f"no annotations parsed from {labels}")
    return collect_items(sorted(references), lambda uri: audio_dir / f"{uri}.wav", references.get)


def _notsofar_word_spans(transcription: list[dict]):
    """``(start, end, speaker)`` of every timed word (pauses inside utterances are not speech)."""
    for utt in transcription:
        for word in utt.get("word_timing") or []:
            if len(word) >= 3 and word[1] is not None and word[2] is not None:
                yield float(word[1]), float(word[2]), utt["speaker_id"]


def load_notsofar(audio_dir: Path, labels: Path) -> list[Item]:
    """NOTSOFAR-1.

    Meetings are laid out as ``MTG_<id>/``, each with its ground truth
    ``gt_transcription.json`` and one ``sc_<device>/ch0.wav`` per single-channel
    far-field device. Every ``(meeting, device)`` pair is a separate item named
    ``<meeting>__<device>``, sharing the meeting's word-level reference. The audio
    directory and the labels directory are both the directory holding ``MTG_*``.
    """
    references, audio = {}, {}
    for trans_path in sorted(labels.glob("MTG_*/gt_transcription.json")):
        meeting = trans_path.parent.name
        with open(trans_path) as f:
            spans = _notsofar_word_spans(json.load(f))
            references[meeting] = annotation_from_spans(meeting, spans)
        for wav in sorted((audio_dir / meeting).glob("sc_*/ch0.wav")):
            if wav.stat().st_size > 0:  # skip incomplete downloads
                audio[f"{meeting}__{wav.parent.name}"] = wav
    if not references:
        raise FileNotFoundError(f"no MTG_*/gt_transcription.json under {labels}")
    return collect_items(
        audio, audio.__getitem__, lambda uri: references[uri.split("__")[0]].copy()
    )


def _parse_ramc_txt(path: Path, uri: str) -> Annotation:
    """Parse a per-conversation TXT file: ``[start,end]<TAB>speaker_id<TAB>...`` lines."""
    spans = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            span, speaker = parts[0].strip(), parts[1].strip()
            if not (span.startswith("[") and span.endswith("]")) or not speaker:
                continue
            if speaker == RAMC_NON_SPEAKER:
                continue
            try:
                start, end = (float(t) for t in span[1:-1].split(","))
            except ValueError:
                continue
            spans.append((start, end, speaker))
    return annotation_from_spans(uri, spans)


def load_ramc(audio_dir: Path, labels: Path) -> list[Item]:
    """MagicData-RAMC.

    Audio: ``<audio_dir>/<uri>.wav`` (``MDT2021S003/WAV``), with the per-conversation
    transcripts in the sibling ``MDT2021S003/TXT/<uri>.txt``. Labels: the partition
    file of the evaluated split (``DataPartition/test.tsv``), listing its wavs.
    """
    txt_dir = audio_dir.parent / "TXT"
    uris = [
        Path(fname).stem
        for line in labels.read_text().splitlines()
        if (fname := line.split("\t")[0].strip()).endswith(".wav")
    ]
    if not uris:
        raise ValueError(f"no .wav entries parsed from {labels}")

    def reference(uri: str) -> Annotation | None:
        txt_path = txt_dir / f"{uri}.txt"
        return _parse_ramc_txt(txt_path, uri) if txt_path.is_file() else None

    return collect_items(uris, lambda uri: audio_dir / f"{uri}.wav", reference)


def load_voxconverse(audio_dir: Path, labels: Path) -> list[Item]:
    """VoxConverse (dev or test).

    Audio: ``<audio_dir>/<uri>.wav``. Labels: ``<labels>/<uri>.rttm`` (the ``dev/``
    or ``test/`` directory of the voxconverse repository).
    """
    return rttm_dir_items(audio_dir, labels, ".wav")


DATASETS = {
    "ami": load_ami,
    "dihard": load_dihard,
    "msdwild": load_msdwild,
    "notsofar": load_notsofar,
    "ramc": load_ramc,
    "voxconverse": load_voxconverse,
}
