"""Gemma 4 hybrid and standalone HQ-family loaders for Hugging Face Transformers.

``load_gemma4_hq2`` is the hybrid path: HQ2/HQ3 matrices live in an archive
while unquantized tensors stream from a local Safetensors checkpoint.
``load_gemma4_hq2_package`` is self-contained: its archive holds those
remaining tensors losslessly as HQ ``RAW`` payloads, and its directory holds
the normal tokenizer/config assets needed by Transformers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .archive import HQ2_FORMAT, HQ3_FORMAT, HQModel, load_model
from .raw import is_raw_descriptor
from .torch_inference import HQ2Linear, HQ3Linear, _require_torch


def _checkpoint_key_to_model_key(name: str) -> str:
    """Map the supplied Gemma 4 checkpoint's legacy vision names to HF 5.12."""
    if name == "model.embed_vision.embedding_projection.weight":
        return "model.embed_vision.multimodal_embedder.embedding_projection.weight"
    if name.startswith("model.vision_embedder."):
        return "model.embed_vision." + name.removeprefix("model.vision_embedder.")
    return name


def _parent_and_attribute(module: Any, dotted_name: str) -> tuple[Any, str]:
    parent = module
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _module_at(module: Any, dotted_name: str) -> Any:
    parent, attribute = _parent_and_attribute(module, dotted_name)
    return getattr(parent, attribute)


def _set_module(module: Any, dotted_name: str, value: Any) -> None:
    parent, attribute = _parent_and_attribute(module, dotted_name)
    setattr(parent, attribute, value)


def _require_transformers():
    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from transformers import AutoConfig, AutoModelForImageTextToText
    except ImportError as exc:
        raise RuntimeError("Gemma HQ2 loading requires transformers and accelerate in addition to Torch") from exc
    return init_empty_weights, set_module_tensor_to_device, AutoConfig, AutoModelForImageTextToText


def _packed_names(archive: HQModel) -> tuple[str, ...]:
    return tuple(
        name for name in archive.tensor_names
        if archive.descriptor(name).format == HQ2_FORMAT or archive.descriptor(name).format == HQ3_FORMAT
    )


def _prepare_gemma4_model(
    archive: HQModel,
    config_dir: Path,
    *,
    device: str,
    dtype: Any | None,
    progress: Callable[[str], None] | None,
):
    runtime = _require_torch()
    init_empty_weights, _, AutoConfig, AutoModelForImageTextToText = _require_transformers()
    if archive.metadata.get("architecture") != "gemma4":
        raise ValueError(f"Archive is not recorded as Gemma 4: {archive.metadata.get('architecture')!r}")
    if not runtime.cuda.is_available() or not runtime.version.hip:
        raise RuntimeError("Gemma HQ loading currently requires a visible ROCm Torch device")
    if dtype is None:
        dtype = runtime.bfloat16
    config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(config)

    packed_names = _packed_names(archive)
    if not packed_names:
        raise ValueError("Gemma HQ archive has no canonical HQ2/HQ3 weights")
    hq2_count = 0
    hq3_count = 0
    for tensor_name in packed_names:
        format = archive.descriptor(tensor_name).format
        if not tensor_name.endswith(".weight"):
            raise ValueError(f"HQ Gemma archive tensor must be a weight: {tensor_name}")
        module_name = tensor_name.removesuffix(".weight")
        original = _module_at(model, module_name)
        if not isinstance(original, runtime.nn.Linear):
            raise TypeError(f"{module_name} is {type(original).__name__}, not nn.Linear")
        if original.bias is not None:
            raise ValueError(f"{module_name} has a bias; HQ Gemma projections must be bias-free")
        tensor = archive.tensor(tensor_name)
        if format == HQ2_FORMAT:
            replacement = HQ2Linear.from_archive(tensor).to(device)
            hq2_count += 1
        else:
            replacement = HQ3Linear.from_archive(tensor).to(device)
            hq3_count += 1
        _set_module(model, module_name, replacement)
    if progress:
        progress(f"Installed {hq2_count} HQ2 and {hq3_count} HQ3 linear layers on {device}.")
    return model, runtime, dtype, packed_names


def _finish_gemma4_model(model: Any, progress: Callable[[str], None] | None, *, package: bool) -> Any:
    # Gemma ties lm_head to embed_tokens; source checkpoints intentionally omit
    # the duplicate lm_head tensor, so tie only after embeddings are resident.
    model.tie_weights()
    unresolved = [
        name for name, value in list(model.named_parameters()) + list(model.named_buffers()) if value.is_meta
    ]
    if unresolved:
        source = "standalone package" if package else "hybrid checkpoint"
        raise RuntimeError(f"Gemma HQ {source} left meta tensors unresolved: {unresolved[:8]}")
    model.eval()
    if progress:
        progress("Gemma 4 HQ model is ready for inference.")
    return model


def load_gemma4_hq2(
    base_checkpoint: str | Path,
    archive_path: str | Path,
    *,
    device: str = "cuda",
    dtype: Any | None = None,
    progress: Callable[[str], None] | None = print,
):
    """Load a hybrid Gemma 4 model with HQ2/HQ3 layers plus a BF16 base checkpoint."""
    try:
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("Hybrid Gemma HQ2 loading requires safetensors and accelerate") from exc
    base_checkpoint = Path(base_checkpoint)
    if base_checkpoint.is_file():
        checkpoint_file = base_checkpoint
        config_dir = base_checkpoint.parent
    else:
        checkpoint_file = base_checkpoint / "model.safetensors"
        config_dir = base_checkpoint
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Expected Gemma Safetensors checkpoint at {checkpoint_file}")
    archive = load_model(archive_path)
    if any(
        archive.descriptor(name).format != HQ2_FORMAT and archive.descriptor(name).format != HQ3_FORMAT
        for name in archive.tensor_names
    ):
        raise ValueError("This archive contains RAW tensors; use load_gemma4_hq2_package() instead")
    model, runtime, dtype, packed_names = _prepare_gemma4_model(
        archive, config_dir, device=device, dtype=dtype, progress=progress
    )
    packed_name_set = set(packed_names)
    model_state = set(model.state_dict())
    loaded = 0
    with safe_open(checkpoint_file, framework="pt", device="cpu") as source:
        for source_name in source.keys():
            if source_name in packed_name_set:
                continue
            destination_name = _checkpoint_key_to_model_key(source_name)
            if destination_name not in model_state:
                raise KeyError(f"Checkpoint tensor {source_name!r} does not map to this Gemma model")
            value = source.get_tensor(source_name)
            if value.is_floating_point() and value.dtype != dtype:
                value = value.to(dtype)
            set_module_tensor_to_device(model, destination_name, device, value=value.to(device))
            loaded += 1
            if progress and (loaded % 32 == 0 or loaded == len(source.keys()) - len(packed_name_set)):
                progress(f"Loaded {loaded} BF16/base tensors to {device}.")
    return _finish_gemma4_model(model, progress, package=False)


def load_gemma4_hq2_package(
    package_dir: str | Path,
    *,
    device: str = "cuda",
    dtype: Any | None = None,
    progress: Callable[[str], None] | None = print,
):
    """Load a standalone Gemma 4 HQ2/HQ3 package with no base checkpoint required."""
    package_dir = Path(package_dir)
    archive_path = package_dir / "model.hq2"
    if not archive_path.is_file() or not (package_dir / "config.json").is_file():
        raise FileNotFoundError(f"Expected standalone package assets in {package_dir}")
    archive = load_model(archive_path)
    if not archive.metadata.get("standalone_package"):
        raise ValueError("Archive is not marked as a standalone HQ package")
    model, runtime, dtype, packed_names = _prepare_gemma4_model(
        archive, package_dir, device=device, dtype=dtype, progress=progress
    )
    raw_names = tuple(name for name in archive.tensor_names if name not in set(packed_names))
    unsupported = [name for name in raw_names if not is_raw_descriptor(archive.descriptor(name).format)]
    if unsupported:
        raise ValueError(f"Standalone package has unsupported non-HQ tensors: {unsupported[:3]}")
    _, set_module_tensor_to_device, _, _ = _require_transformers()
    model_state = set(model.state_dict())
    loaded = 0
    for source_name in raw_names:
        destination_name = _checkpoint_key_to_model_key(source_name)
        if destination_name not in model_state:
            raise KeyError(f"Package tensor {source_name!r} does not map to this Gemma model")
        stored = archive.raw_tensor(source_name)
        try:
            value = stored.to_torch()
            if value.is_floating_point() and value.dtype != dtype:
                value = value.to(dtype)
            set_module_tensor_to_device(model, destination_name, device, value=value.to(device))
        finally:
            stored.close()
        loaded += 1
        if progress and (loaded % 32 == 0 or loaded == len(raw_names)):
            progress(f"Loaded {loaded}/{len(raw_names)} lossless package tensors to {device}.")
    return _finish_gemma4_model(model, progress, package=True)


__all__ = ["load_gemma4_hq2", "load_gemma4_hq2_package"]
