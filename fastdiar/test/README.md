# Evaluation

`evaluate.py` streams every file of a diarization benchmark through the streaming diarizer and
reports the DER computed by `pyannote.metrics` (collar 0, no UEM; overlapped speech is scored
unless `--skip-overlap` is given).

`eval_vox1.py` measures the speaker-verification EER of the
models on VoxCeleb1 (see [Speaker verification](#speaker-verification-voxceleb1)), and
`measure_rtf.py` the diarizer's real-time factor (see [Real-time factor](#real-time-factor)).

## Datasets

Each dataset is passed to the script as an audio directory (`--audio-dir`) and a labels path
(`--labels`). The paths below are relative to wherever you downloaded the dataset.

### AMI (IHM-MIX): `ami`

- Audio: [FluidInference/ami-corpus-mirror](https://huggingface.co/datasets/FluidInference/ami-corpus-mirror),
```
hf download FluidInference/ami-corpus-mirror --repo-type dataset --local-dir ami-corpus-mirror
```
- Labels: `git clone https://github.com/pyannote/AMI-diarization-setup`

The 16 meetings of the official AMI test split (9.1 h, 3–4 speakers each). The audio is the
`Mix-Headset` sum of the close-talking microphones (IHM-MIX), so results are not comparable to
distant-microphone (SDM) numbers. The labels are the manual `only_words` annotations.

```
--audio-dir ami-corpus-mirror/sdm --labels AMI-diarization-setup/only_words/rttms/test
```

### DIHARD III: `dihard`

- [DIHARD III](https://dihardchallenge.github.io/dihard3/), licensed from the LDC: evaluation set LDC2022S14.

Recordings from 11 domains, including clinical interviews, courtroom, restaurant, meetings and
web video. Scoring whole files with collar 0 matches the DIHARD III "full" protocol.

```
--audio-dir third_dihard_challenge_eval/data/flac --labels third_dihard_challenge_eval/data/rttm
```

### MSDWild: `msdwild`

- [X-LANCE/MSDWILD](https://github.com/X-LANCE/MSDWILD): download the audio archive linked in
  the README; the labels are the RTTM files in the repository's `rttms/` directory.

In-the-wild video clips. The `few` validation protocol has 490 clips (9.9 h, 2–4 speakers);
pass `many.val.rttm` instead for the crowded-scene protocol.

```
--audio-dir MSDWild/wav --labels MSDWILD/rttms/few.val.rttm
```

### NOTSOFAR-1: `notsofar`

- [microsoft/NOTSOFAR1-Challenge](https://github.com/microsoft/NOTSOFAR1-Challenge): download a
  meeting subset with the download script described in the repository README.

Real meetings recorded by several far-field devices at once. Each single-channel device
(`MTG_*/sc_*/ch0.wav`) is scored as a separate file against the meeting's word-level
reference (`MTG_*/gt_transcription.json`), so audio and labels are the same directory. For
example, 129 meetings give 819 recordings (84 h, 3–7 speakers).

```
--audio-dir notsofar --labels notsofar
```

### MagicData-RAMC: `ramc`

- [OpenSLR 123](https://www.openslr.org/123/): download and extract the corpus, which contains
  `MDT2021S003/` (`WAV/`, `TXT/`) and `DataPartition/`.

Mandarin two-party conversations. The labels are the split's partition file (the test split
has 43 conversations, 20.6 h); the references are read from the `TXT/` directory next to
`WAV/`.

```
--audio-dir RAMC/MDT2021S003/WAV --labels RAMC/DataPartition/test.tsv
```

### VoxConverse: `voxconverse`

- [joonson/voxconverse](https://github.com/joonson/voxconverse): clone it for the labels
  (`dev/` and `test/` RTTM directories) and download the dev and test audio archives linked in
  the README.

Multi-speaker clips from YouTube debates, news and talk shows. The test set has 232 files
(43.5 h, 1–21 speakers).

```
--audio-dir voxconverse/wav/test --labels voxconverse/test
```

## Usage

Run from the repository root:

```bash
python -m fastdiar.test.evaluate voxconverse \
    --audio-dir voxconverse/wav/test --labels voxconverse/test \
    -m large -o out/voxconverse
```

- `-m {small,medium,large}`: model size (default `large`), downloaded on first use, or a local
  `.pt` checkpoint file.
- `-o DIR`: save the hypothesis of every file as `DIR/<uri>.rttm`. Files that already have one
  are not diarized again, so an interrupted run resumes where it stopped.
- `--skip-overlap`: exclude regions where several reference speakers overlap from scoring.
  Without it, overlapped speech is scored.
- `-j N`: number of parallel processes, each diarizing whole files on one CPU thread (default:
  one per CPU). `-j 1` runs everything in the main process.

Files are diarized on the CPU, with a tqdm progress bar. The script then
prints the accumulated DER row of the pyannote report. When a dataset has files with 5 or more
speakers, it prints three rows: files with at most 4 speakers, files with 5 or more, and all
files.

## Results

DER (%) of FASTDIAR (the `large` model, 960 ms latency) on the test sets described in
[Datasets](#datasets), with collar 0 and overlapped speech excluded from scoring (`--skip-overlap`):

| Model | NOTSOFAR | VoxConverse | DIHARD III | AMI | MSDWild | RAMC |
| --- | --- | --- | --- | --- | --- | --- |
| FASTDIAR (`large`) | 21.58 | 12.35 | 26.06 | 22.15 | 24.69 | 21.61 |

The real-time factor of the whole diarizer (encoder, VAD and clustering) is **0.19** on one thread of
an AMD Threadripper PRO 5995WX CPU.

## Speaker verification (VoxCeleb1)

`eval_vox1.py` needs the [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
test set audio (`--audio-dir`: the `wav/` directory the trial paths are relative to) and the
cleaned trial list `veri_test2.txt` (`--protocol`).

```bash
python -m fastdiar.test.eval_vox1 --audio-dir vox1/test/wav --protocol vox1/test/veri_test2.txt -m large
```

- `-m {b6,small,medium,large}`: `b6` is the released redimnet2-b6-vb2+vox2_v0-lm model (one
  embedding per utterance, weights from torch hub); the others are the streaming encoder (one
  embedding per frame).
- `--enroll-model b6`: embed the enrollment side with another model (cross-model scoring).
- `--cache [-o DIR]`: save and reuse float16 embeddings, next to the audio or under `DIR`.
- `--shift-sec S`: audio block fed to the streaming encoder (default 0.32).
- `--skip-frames N`: streaming frames left out of scoring at the start of every utterance
  (default 12, i.e. 960 ms).
- `-j N`: number of parallel embedding processes (default: one per CPU).

## Real-time factor

`measure_rtf.py` diarizes the first 5 files of the [VoxConverse dev](https://github.com/joonson/voxconverse) set, each cut to its first
minute, on a single CPU thread, and prints the real-time factor (processing time / audio duration)
of every file and of all of them:

```bash
python -m fastdiar.measure_rtf voxconverse/wav/dev -m small
```
