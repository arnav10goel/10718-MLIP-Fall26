"""Frame features for the checkpoint baselines."""

import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_clip_encoder(checkpoint_path: str | Path, *, device: str = "cpu") -> tuple:
    """Load a local ViT-B-32 checkpoint and return model, device, metadata.

    Require a nonempty local .pt/.pth/.bin state-dict or .safetensors file.
    Expand ~ and resolve relative paths from the caller's directory. Build
    ViT-B-32 with QuickGELU in float32 on CPU, load its full state strictly,
    then move to the explicit device and enter evaluation mode. Invalid
    paths/formats/device syntax raise ValueError; checkpoint restoration and
    unavailable-device errors propagate, without random-weight fallback.

    No remote model identifiers, downloads, cache selection or automatic
    device selection occur. Use OpenCLIP's weights-only restoration; legacy
    TorchScript archives need conversion before using this loader. The file
    must stay unchanged during loading. Record its pre-load hash, actual
    device, package versions and the encode_clip_frames configuration.
    A matching architecture and hash do not establish that weights are the
    official OpenAI release. Torch and OpenCLIP are imported only on use.
    """
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be an explicit Torch device string")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError("Checkpoint must be a nonempty local regular file")
    if checkpoint.suffix not in (".pt", ".pth", ".bin", ".safetensors"):
        raise ValueError("Checkpoint must be a state-dict or safetensors file")

    import open_clip
    import torch

    try:
        requested_device = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError("device must be a valid Torch device string") from error
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    size_bytes = checkpoint.stat().st_size
    model = open_clip.create_model(
        "ViT-B-32", pretrained=None, precision="fp32", device="cpu",
        force_quick_gelu=True, pretrained_image=False, pretrained_text=False,
    )
    open_clip.load_checkpoint(
        model, str(checkpoint), strict=True, weights_only=True, device="cpu",
    )
    model = model.to(requested_device).eval()
    actual_device = str(next(model.parameters()).device)
    metadata = {
        "architecture": "ViT-B-32",
        "force_quick_gelu": True,
        "precision": "fp32",
        "device": actual_device,
        "requested_device": str(requested_device),
        "checkpoint": {"path": str(checkpoint), "sha256": digest, "size_bytes": size_bytes},
        "runtime": {"torch": version("torch"), "open_clip": version("open-clip-torch")},
        "preprocessing": {
            "frame_size": [224, 224], "input_color": "BGR", "model_color": "RGB",
            "pixel_scale": 1 / 255, "mean": list(CLIP_MEAN), "std": list(CLIP_STD),
            "resize": "caller", "crop": None, "embedding_normalization": "L2",
        },
    }
    return model, actual_device, metadata


def encode_clip_frames(model, device, frames_bgr: np.ndarray, *, batch_size: int = 32) -> np.ndarray:
    """Encode sampled BGR frames with a caller-owned CLIP image encoder.

    Require nonempty uint8 (N,224,224,3) frames and an already loaded model
    in evaluation mode on the supplied device. Preserve the teammate's
    preprocessing: BGR to RGB, NCHW float pixels divided by 255, then CLIP
    channel normalization. Frames are already resized by the sampler; this
    function does not crop or resize them. Process at most batch_size frames
    on the device at once, in inference mode, without changing model mode.

    Return ordered CPU float32 (N,D) unit vectors. Encoder outputs must be
    floating tensors with consistent width, finite values and nonzero rows.
    Scale each row before its float32 norm to avoid overflow or underflow.
    Invalid inputs or outputs raise ValueError; device and
    encoder exceptions propagate. Read-only and strided input arrays work.
    No model loading, downloads, file access, timestamps or step inference
    occur here. The caller establishes encoder/checkpoint identity and must
    use the same encoder for both videos. Torch is an optional dependency.
    """
    if (
        not isinstance(frames_bgr, np.ndarray)
        or frames_bgr.dtype != np.uint8
        or frames_bgr.ndim != 4
        or frames_bgr.shape[0] == 0
        or frames_bgr.shape[1:] != (224, 224, 3)
    ):
        raise ValueError("Frames must be a nonempty uint8 (N,224,224,3) BGR array")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if model.training:
        raise ValueError("CLIP model must already be in evaluation mode")

    import torch

    parts = []
    feature_width = None
    with torch.inference_mode():
        mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        for start in range(0, len(frames_bgr), batch_size):
            rgb = np.ascontiguousarray(frames_bgr[start:start + batch_size, :, :, ::-1])
            batch = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255.0)
            batch = (batch.to(device) - mean) / std
            features = model.encode_image(batch)
            if (
                not isinstance(features, torch.Tensor)
                or not features.is_floating_point()
                or features.ndim != 2
                or features.shape[0] != len(batch)
                or features.shape[1] == 0
                or (feature_width is not None and features.shape[1] != feature_width)
            ):
                raise ValueError("Encoder must return a floating (batch,D) tensor with consistent width")
            if not torch.isfinite(features).all():
                raise ValueError("Encoder features must contain finite values")
            scale = features.abs().amax(dim=-1, keepdim=True)
            if (scale <= 0).any():
                raise ValueError("Encoder feature rows must be nonzero")
            features = (features / scale).float()
            norms = features.norm(dim=-1, keepdim=True)
            feature_width = features.shape[1]
            parts.append((features / norms).cpu().numpy())
    return np.concatenate(parts).astype(np.float32, copy=False)
