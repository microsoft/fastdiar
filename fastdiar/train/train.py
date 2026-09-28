"""Distillation training of a streaming ReDimNet2 (see ``configs/train.yaml``).

Usage:
    python -m fastdiar.train.train --config configs/train.yaml --data-root /data \\
        [--override key=value ...]

Dataset paths in the config are relative to ``--data-root`` (absolute ones are
used as given). Every ``--override`` is one ``dotted.key=value``, with the value
parsed as YAML, e.g. ``--override datasets.voxceleb.weight=0.5``.
"""

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import lightning as L  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import TensorBoardLogger  # noqa: E402
from lightning.pytorch.utilities import rank_zero_only  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fastdiar.train.dataset import (  # noqa: E402
    LibriHeavyMixDataset,
    RandomBatchSampler,
    RandomWavSource,
    ReverbNoiseAugment,
    VoxCelebDataset,
    WeightedConcatDataset,
    collate_fn,
)
from fastdiar.train.trainer import DistillTrainer, Vox1EERCallback  # noqa: E402


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value: {override!r}")
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def build_dataloader(cfg: dict[str, Any], root: Path) -> DataLoader:
    """The weighted VoxCeleb2 + LibriHeavyMix mixture."""
    shared = {
        "target_duration": cfg["target_duration"],
        "sample_rate": cfg["sample_rate"],
        "max_speakers": cfg["max_speakers"],
    }
    vox = dict(cfg["datasets"]["voxceleb"])
    aug = vox.pop("augment")
    musan = root / aug["musan_dir"]
    augment = ReverbNoiseAugment(
        reverb_source=RandomWavSource(root / aug["rirs_dir"]),
        noise_source=RandomWavSource({kind: musan / kind for kind in ("noise", "music", "speech")}),
        sample_rate=cfg["sample_rate"],
        aug_prob=aug["aug_prob"],
    )
    lhm = dict(cfg["datasets"]["libriheavymix"])
    datasets = {
        "voxceleb": VoxCelebDataset(
            parquet_path=root / vox.pop("parquet_path"),
            audio_dir=root / vox.pop("audio_dir"),
            augment=augment,
            **{k: v for k, v in vox.items() if k != "weight"},
            **shared,
        ),
        "libriheavymix": LibriHeavyMixDataset(
            parquet_path=root / lhm.pop("parquet_path"),
            mixture_dir=root / lhm.pop("mixture_dir"),
            source_dir=root / lhm.pop("source_dir"),
            **{k: v for k, v in lhm.items() if k != "weight"},
            **shared,
        ),
    }
    weights = [cfg["datasets"][name]["weight"] for name in datasets]
    for (name, dataset), weight in zip(datasets.items(), weights, strict=True):
        print(f"[train] {name}: {len(dataset)} items, weight {weight}")
    combined = WeightedConcatDataset(list(datasets.values()), weights)
    sampler = RandomBatchSampler(
        batch_size=cfg["batch_size"],
        num_batches_per_epoch=cfg["num_batches_per_epoch"],
        seed=cfg["seed"],
    )
    return DataLoader(
        combined,
        batch_sampler=sampler,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=cfg["pin_memory"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML training config")
    parser.add_argument(
        "--data-root", type=Path, required=True, help="directory the dataset paths are relative to"
    )
    parser.add_argument(
        "--override", action="append", default=[], help="dotted.key=value (repeatable)"
    )
    args = parser.parse_args()
    cfg = apply_overrides(yaml.safe_load(args.config.read_text()), args.override)

    torch.set_float32_matmul_precision("medium")
    L.seed_everything(cfg["seed"])

    pl_module = DistillTrainer(cfg)
    train_dataloader = build_dataloader(cfg, args.data_root)

    log_dir = Path(cfg["log_dir"])
    if rank_zero_only.rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    logger = TensorBoardLogger(save_dir=log_dir, default_hp_metric=False)
    vox1 = cfg["vox1_eval"]
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=logger.log_dir,
            filename="{epoch}-{step}",
            save_weights_only=cfg["save_weights_only"],
        ),
        Vox1EERCallback(
            protocol=args.data_root / vox1["protocol"],
            wav_dir=args.data_root / vox1["wav_dir"],
            skip_frames=vox1["skip_frames"],
        ),
    ]
    trainer = L.Trainer(
        accelerator=cfg["accelerator"],
        devices=cfg["devices"],
        precision=cfg["precision"],
        logger=logger,
        max_epochs=cfg["max_epochs"],
        log_every_n_steps=cfg["log_every_n_steps"],
        gradient_clip_val=cfg["gradient_clip_val"],
        callbacks=callbacks,
        # Every process draws its own batches (see RandomBatchSampler).
        use_distributed_sampler=False,
    )
    trainer.fit(pl_module, train_dataloader)


if __name__ == "__main__":
    main()
