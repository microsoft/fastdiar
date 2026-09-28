"""Lightning module and callback for distilling a streaming ReDimNet2 from a frozen b6.

The frozen teacher (the released b6, from torch hub) embeds every speaker's
clean crop; the streaming student is trained so that each of its per-frame
embeddings of the (mixed, augmented) input matches the teacher embedding of the
speaker active in that frame, by cosine similarity.

The teacher's parameters are frozen, but it stays in ``train()`` mode so its
BatchNorm running statistics keep following the input distribution; its
forward runs under ``no_grad``.
"""

import json
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from fastdiar.cli import load_audio
from fastdiar.encoder import StreamingReDimNet2
from fastdiar.model.redimnet2 import load_hub_model
from fastdiar.test.eval_vox1 import compute_eer, read_protocol, utterance_vector


def _load_state_dict(src: str) -> dict[str, torch.Tensor]:
    """State dict of a torch hub URL, or of a local (Lightning) checkpoint."""
    if src.startswith(("http://", "https://")):
        checkpoint = torch.hub.load_state_dict_from_url(src, map_location="cpu", weights_only=True)
        return checkpoint["state_dict"]
    obj = torch.load(src, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj)
    # A DistillTrainer checkpoint saves the student under `student.`.
    return {k.removeprefix("student."): v for k, v in state.items()}


def _build_student(config_path: str, ckpt: str | None) -> StreamingReDimNet2:
    """The streaming student, initialized from ``ckpt`` where the shapes allow.

    A tensor whose shape differs only in a smaller last dimension (e.g. the
    projection head, which has no pooling in the student) gets the leading
    slice of the checkpoint's; any other mismatch keeps its random init.
    """
    with open(config_path) as f:
        model = StreamingReDimNet2(**json.load(f))
    if ckpt is None:
        return model
    state = _load_state_dict(ckpt)
    # Rename legacy `stage{i}.{pos}` keys first, so they can be matched below.
    model.backbone._remap_legacy_state_dict(state, "backbone.")
    target = model.state_dict()
    loaded, sliced, skipped = {}, [], []
    for key, value in state.items():
        want = target.get(key)
        if want is None or want.shape == value.shape:
            loaded[key] = value
        elif value.shape[:-1] == want.shape[:-1] and value.shape[-1] >= want.shape[-1]:
            loaded[key] = value[..., : want.shape[-1]].contiguous()
            sliced.append(key)
        else:
            skipped.append(key)
    result = model.load_state_dict(loaded, strict=False)
    print(
        f"[DistillTrainer] student from {ckpt}: {len(result.missing_keys)} missing, "
        f"{len(result.unexpected_keys)} unexpected, {len(sliced)} last-dim-sliced, "
        f"{len(skipped)} shape-mismatched keys"
    )
    for name, keys in (
        ("missing", result.missing_keys),
        ("unexpected", result.unexpected_keys),
        ("last-dim-sliced", sliced),
        ("shape-mismatched", skipped),
    ):
        if keys:
            print(f"  {name} (first 5): {keys[:5]}")
    return model


class LinearWarmupDecayScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Linear warmup from ``initial_lr`` to ``final_lr`` over ``warmup_steps``,
    then linear decay back to ``initial_lr`` over the remaining steps."""

    def __init__(self, optimizer, warmup_steps, total_steps, initial_lr, final_lr):
        warmup_steps = max(1, int(warmup_steps))
        total_steps = max(warmup_steps + 1, int(total_steps))
        self.lrs = torch.cat(
            (
                torch.linspace(initial_lr, final_lr, warmup_steps),
                torch.linspace(final_lr, initial_lr, total_steps - warmup_steps + 1),
            )
        )
        super().__init__(optimizer)

    def get_lr(self):
        step = min(max(self.last_epoch, 0), len(self.lrs) - 1)
        return [float(self.lrs[step]) for _ in self.base_lrs]


class DistillTrainer(L.LightningModule):
    """Distills the streaming student (``student_config``) from the frozen b6 teacher."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config

        self.teacher = load_hub_model(config["teacher_ckpt"])
        self.student = _build_student(config["student_config"], config.get("student_ckpt"))
        if self.teacher.linear.out_features != self.student.linear.out_features:
            raise ValueError("teacher and student embedding sizes differ")
        for p in self.teacher.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.train(True)  # keep updating its BatchNorm statistics
        return self

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # The teacher is frozen and restored from torch hub; do not save it.
        state = checkpoint["state_dict"]
        for key in [k for k in state if k.startswith("teacher.")]:
            del state[key]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint["state_dict"]
        for key, value in self.teacher.state_dict().items():
            state.setdefault(f"teacher.{key}", value)

    def training_step(self, batch, batch_idx):
        student_audio = batch["student_audio"]  # (B, T)
        teacher_audio = batch["teacher_audio"]  # (B, S, T)
        seg_label = batch["seg_label"]  # (B, T) int8: speaker slot, -1 ignored
        speaker_valid = batch["speaker_valid"]  # (B, S) bool
        B, S, T = teacher_audio.shape

        # Teacher: one batched forward over the valid speaker crops only.
        with torch.no_grad():
            idx = torch.nonzero(speaker_valid.reshape(-1)).flatten()
            crops = teacher_audio.reshape(B * S, T).index_select(0, idx)
            # BatchNorm in train mode needs more than one sample.
            emb = self.teacher(crops.repeat(2, 1) if len(idx) == 1 else crops)[: len(idx)]
            t_emb = emb.new_zeros(B * S, emb.shape[1]).index_copy_(0, idx, emb)
            t_emb = F.normalize(t_emb.reshape(B, S, -1), dim=-1, eps=1e-8)

        s_emb = F.normalize(self.student(student_audio), dim=-1, eps=1e-8)  # (B, T_fr, D)
        cos_all = torch.einsum("btd,bsd->bts", s_emb, t_emb)  # (B, T_fr, S)
        target, valid = self._frame_targets(seg_label, s_emb.shape[1], S)
        cos = cos_all.gather(2, target.unsqueeze(-1)).squeeze(-1)  # (B, T_fr)

        valid = valid.to(cos.dtype)
        denom = valid.sum().clamp_min(1.0)
        loss = ((1.0 - cos) * valid).sum() / denom
        # Averaged over the processes of a distributed run.
        log = {"prog_bar": True, "on_step": True, "on_epoch": True, "sync_dist": True}
        self.log("train_loss", loss, **log)
        self.log("train_cos", (cos * valid).sum() / denom, **log)
        return loss

    @staticmethod
    def _frame_targets(seg_label: torch.Tensor, n_frames: int, num_speakers: int):
        """Per-frame target speaker slot ``(B, n_frames)`` and validity mask.

        A frame is a valid target only if a single speaker is present in all
        the samples it covers and none of them is ignored (``-1``).
        """
        seg = seg_label.float().unsqueeze(1)  # (B, 1, T)

        def present(value):
            return F.adaptive_max_pool1d((seg == value).float(), n_frames).squeeze(1) > 0.5

        has = torch.stack([present(s) for s in range(num_speakers)], dim=-1)  # (B, T_fr, S)
        target = has.float().argmax(dim=-1)
        valid = (has.sum(dim=-1) == 1) & ~present(-1)
        return target, valid

    def configure_optimizers(self):
        cfg = self.config
        optimizer = AdamW(self.student.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = total_steps * cfg["warmup_epochs"] // cfg["max_epochs"]
        scheduler = LinearWarmupDecayScheduler(
            optimizer, warmup_steps, total_steps, initial_lr=cfg["initial_lr"], final_lr=cfg["lr"]
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class Vox1EERCallback(L.Callback):
    """Logs the student's VoxCeleb1 EER (``vox1_eer``, %) after every training epoch.

    Scores like ``fastdiar.test.eval_vox1``: the mean of an utterance's
    per-frame embeddings after its first ``skip_frames``. In a distributed run
    every process embeds its share of the files, with the BatchNorm statistics
    of rank 0, and rank 0 computes the EER.
    """

    def __init__(self, protocol: Path, wav_dir: Path, skip_frames: int):
        self.wav_dir = Path(wav_dir)
        self.skip_frames = int(skip_frames)
        labels, self.enrolls, self.trials = read_protocol(Path(protocol))
        self.labels = np.asarray(labels)
        self.files = sorted(set(self.enrolls) | set(self.trials))

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: DistillTrainer) -> None:
        student = pl_module.student
        distributed = trainer.world_size > 1
        if distributed:
            # BatchNorm statistics are the only buffers training changes.
            for module in student.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    for buffer in module.buffers(recurse=False):
                        dist.broadcast(buffer, src=0)
        student.eval()
        vectors = {}
        for rel in self.files[trainer.global_rank :: trainer.world_size]:
            wav = load_audio(self.wav_dir / rel).to(pl_module.device)
            emb = F.normalize(student(wav[None])[0].float(), dim=-1).cpu().numpy()
            vectors[rel] = utterance_vector(emb, rel, self.skip_frames)
        student.train()
        if distributed:
            shares = [None] * trainer.world_size
            dist.all_gather_object(shares, vectors)
            vectors = {rel: vec for share in shares for rel, vec in share.items()}
        if not trainer.is_global_zero:
            return
        scores = np.array(
            [vectors[e] @ vectors[t] for e, t in zip(self.enrolls, self.trials, strict=True)]
        )
        eer, _ = compute_eer(scores, self.labels)
        trainer.logger.log_metrics({"vox1_eer": eer * 100}, step=trainer.global_step)
        print(f"[Vox1EERCallback] epoch {trainer.current_epoch}: EER {eer * 100:.3f} %")
