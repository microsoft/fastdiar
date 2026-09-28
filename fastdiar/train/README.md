# Training

Distills the streaming encoder (`configs/large.json`) from the frozen, whole-utterance
[ReDimNet2 b6](https://github.com/PalabraAI/redimnet2) model: every per-frame embedding of the
student is trained to match the teacher's embedding of the speaker active in that frame. The
training data is a weighted mix of VoxCeleb2 (single speakers and two-speaker mixtures built on the
fly, with reverb and noise) and LibriHeavyMix (pre-rendered meetings of 2–3 speakers). The student's
VoxCeleb1-test EER is logged after every epoch.

Training needs one or more GPUs. The steps below use `$DATA` for the directory that
holds all the datasets; it is passed to the training script as `--data-root`.

## 1. Download the data

| Dataset | Used for | Where it goes under `$DATA` |
| --- | --- | --- |
| [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) dev | training | `voxceleb/vox2/dev/aac/<speaker>/<video>/<utt>.m4a` |
| [LibriHeavyMix-medium](https://huggingface.co/datasets/zrjin/LibriheavyMix-medium) | training | `LibriheavyMix/` (see below) |
| [MUSAN](https://openslr.org/17) | noise augmentation | `musan/{noise,music,speech}/` |
| [RIRS_NOISES](https://openslr.org/28) | reverb augmentation | `RIRS_NOISES/simulated_rirs/` |
| [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) test | validation (EER) | `voxceleb/vox1/wav/` and `voxceleb/vox1/veri_test2.txt` |

For VoxCeleb1 you only need the test set and the
cleaned trial list
[`veri_test2.txt`](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt).

LibriHeavyMix is split into `audio.tar.gz*` (the mixtures) and `src.tar.gz*` (every speaker's clean
source), about 560 GB in all, and as much again once extracted. Download it and extract both
archives inside `$DATA/LibriheavyMix`:

```bash
hf download zrjin/LibriheavyMix-medium --repo-type dataset --local-dir $DATA/LibriheavyMix
cd $DATA/LibriheavyMix
cat audio.tar.gz?? | tar xz   # -> audio/medium_mtt/<id>.flac
cat src.tar.gz?? | tar xz     # -> src/medium_mtt/<id>/<i>.flac
```

The training protocol is `medium-mtt-lhotse/lsheavymix_cuts_medium.jsonl.gz`. The archives also
hold another subset (`medium`, protocol in `medium-lhotse/`), which is not used.

## 2. Prepare the datasets

`prepare_dataset.py` runs the silero VAD of the diarizer over every file (in parallel, one process
per CPU by default, `-j N` to change it) and writes the parquet protocols the training reads:

```bash
python -m fastdiar.train.prepare_dataset voxceleb \
    --audio-dir $DATA/voxceleb/vox2/dev/aac \
    -o $DATA/voxceleb/vox2/train_meta_vad.parquet

python -m fastdiar.train.prepare_dataset libriheavymix \
    --protocol $DATA/LibriheavyMix/medium-mtt-lhotse/lsheavymix_cuts_medium.jsonl.gz \
    --source-dir $DATA/LibriheavyMix/src/medium_mtt \
    -o $DATA/LibriheavyMix/train_vad_protocol.parquet
```

- **VoxCeleb2**: the speech segments of every utterance; the speaker is the top-level directory.
  Files without speech, or that fail to decode, are left out.
- **LibriHeavyMix**: mixtures with 2 or 3 speakers where a single speaker is active in 20–90% of the
  time. The VAD runs on the clean sources, and the mixtures with more than 5 s of speech are kept.

## 3. Update the training config

In [`configs/train.yaml`](../../configs/train.yaml), every dataset path is relative to
`--data-root`, laid out as in step 1; change a path if your copy is elsewhere (an absolute path is
used as given). Also check:

- `devices`: the GPUs to train on, as a list of indices (`[0, 1, 2, 3]`) or a count (`4`). With
  more than one, training runs with DDP: every GPU draws its own batches, so `batch_size` and
  `num_batches_per_epoch` are per GPU (the effective batch is `batch_size` times the number of
  GPUs), and the VoxCeleb1 evaluation is split across them;
- `log_dir`: where the checkpoints and TensorBoard logs go (default `exp/distill_large`);
- `batch_size` and `num_workers` (per GPU) for your GPU memory and CPU count.

The teacher, and the student's initial weights, are the released b6 checkpoint, downloaded from ReDimNet2
torch hub on the first run.

## 4. Train

```bash
python -m fastdiar.train.train --config configs/train.yaml --data-root $DATA
```

Any config value can be changed from the command line with `--override dotted.key=value`, e.g.
`--override batch_size=16 --override datasets.voxceleb.weight=0.5`. The run writes
`<log_dir>/resolved_config.yaml` and, in `<log_dir>/lightning_logs/version_<N>/`, the checkpoint of
the last epoch and TensorBoard logs of `train_loss`, `train_cos` (mean cosine to the teacher),
the learning rate and `vox1_eer` (%).

To use a trained student with the diarizer, save it as a checkpoint of the inference format: the
student's config and its weights without the `student.` prefix.

```python
import json

import torch

state = torch.load("epoch=99-step=200000.ckpt", map_location="cpu")["state_dict"]
torch.save(
    {
        "model_config": json.load(open("configs/large.json")),
        "state_dict": {k.removeprefix("student."): v for k, v in state.items()},
    },
    "my-model.pt",
)
```

Then pass the file as the model of the inference scripts (`-m my-model.pt`), of
`load_streaming_model("my-model.pt")`, or as `student_ckpt` to continue training from it.
