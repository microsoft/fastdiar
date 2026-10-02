# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""torch.hub entrypoints of the streaming models and the diarizer.

    model = torch.hub.load("microsoft/fastdiar", "large")
    diarizer = torch.hub.load("microsoft/fastdiar", "diarizer", size="small")

The released checkpoints are downloaded to the torch hub cache on first use.
Every entrypoint also takes a local checkpoint, e.g. with a local clone:

    model = torch.hub.load(
        ".", "large", source="local", checkpoint="checkpoints/fastdiar-large-f6d71444.pt"
    )
"""

dependencies = ["torch", "numpy", "scipy", "silero_vad"]

from fastdiar.diarizer import StreamingDiarizer  # noqa: E402
from fastdiar.encoder import StreamingReDimNet2, load_streaming_model  # noqa: E402


def small(checkpoint: str | None = None) -> StreamingReDimNet2:
    """Streaming encoder, 2.4 M parameters (or the local ``checkpoint`` file)."""
    return load_streaming_model(checkpoint or "small")


def medium(checkpoint: str | None = None) -> StreamingReDimNet2:
    """Streaming encoder, 4.6 M parameters (or the local ``checkpoint`` file)."""
    return load_streaming_model(checkpoint or "medium")


def large(checkpoint: str | None = None) -> StreamingReDimNet2:
    """Streaming encoder, 10.4 M parameters (or the local ``checkpoint`` file)."""
    return load_streaming_model(checkpoint or "large")


def diarizer(size: str = "large", checkpoint: str | None = None, **kwargs) -> StreamingDiarizer:
    """Streaming diarizer of the model ``size`` (or the local ``checkpoint`` file).

    ``kwargs`` go to :class:`fastdiar.diarizer.StreamingDiarizer`. (Not ``model``:
    ``torch.hub.load`` takes that name for the entrypoint.)
    """
    return StreamingDiarizer(load_streaming_model(checkpoint or size), **kwargs)
