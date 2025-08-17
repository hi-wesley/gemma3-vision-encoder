# extract_gemma3_vision.py
#
# One-shot extractor for Gemma 3's vision encoder (SigLIP-family) and the Gemma projector.
# - Saves the vision encoder as a standalone Hugging Face model folder (config + safetensors).
# - Saves the image processor used by Gemma.
# - Saves the projector weights (state_dict) plus a small JSON metadata file.
# - Optionally pushes the folder to the Hugging Face Hub.
#
# Usage (CPU-only example):
#   python extract_gemma3_vision.py --out gemma3-vision-extracted
#
# With push to Hub (after `huggingface-cli login`):
#   python extract_gemma3_vision.py --out gemma3-vision-extracted \
#       --push-to-hub --repo-id <your-username>/gemma3-vision-extracted
#
# Notes:
# - You must have accepted the Gemma license on the google/gemma-3-4b-pt model card.
# - This script avoids multiline triple-quoted strings to prevent quoting issues on Windows shells.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    SiglipVisionModel,
)

# Optional (only needed if you pass --push-to-hub)
try:
    from huggingface_hub import HfApi, create_repo, upload_folder  # type: ignore
except Exception:
    HfApi = None  # type: ignore


def _dtype_from_str(name: str):
    name = name.lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _find_vision_module(model) -> SiglipVisionModel:
    # Try common attribute names first.
    for attr in ("vision_tower", "vision_model"):
        if hasattr(model, attr):
            mod = getattr(model, attr)
            if isinstance(mod, SiglipVisionModel):
                return mod
    # Fallback: scan modules for the first SiglipVisionModel
    for m in model.modules():
        if isinstance(m, SiglipVisionModel):
            return m
    raise RuntimeError("Could not locate a SiglipVisionModel inside the Gemma 3 checkpoint.")


def _safe_make_readme(
    model_id: str,
    out_dir: Path,
    vision_hidden: Optional[int],
    llm_hidden: Optional[int],
    num_image_tokens: Optional[int],
    projector_fqn: Optional[str],
):
    """Write a README.md without triple-quoted strings (safer on Windows)."""
    lines = []
    lines += [
        "---",
        "license: gemma",
        "tags:",
        "- image-feature-extraction",
        "- siglip",
        f"base_model: {model_id}",
        "library_name: transformers",
        "---",
        "",
        "# Gemma 3 Vision Encoder (extracted)",
        "",
        f"This repository contains the SigLIP-family vision encoder extracted from **{model_id}**.",
        "It also includes the Gemma multimodal projector weights (state_dict) and a small metadata file.",
        "",
        "## Contents",
        "",
        "- `config.json`, `model.safetensors`: the SigLIP vision encoder",
        "- `preprocessor_config.json`: the image processor settings used by Gemma 3",
        "- `projector_state_dict.pt`: PyTorch state dict for the Gemma projector",
        "- `projector_config.json`: metadata (class, dims, token count if detected)",
        "- `NOTICE`: Gemma Terms pointer",
        "",
        "## Basic usage (encoder as feature extractor)",
        "",
        "```python",
        "from transformers import SiglipVisionModel, AutoImageProcessor",
        "from PIL import Image",
        "import torch",
        "",
        'repo_id = "<your-username>/<your-repo>"',
        "encoder = SiglipVisionModel.from_pretrained(repo_id).eval()",
        "processor = AutoImageProcessor.from_pretrained(repo_id)",
        "",
        'img = Image.open("test.jpg").convert("RGB")',
        "inputs = processor(images=img, return_tensors='pt')",
        "with torch.no_grad():",
        "    feats = encoder(**inputs).last_hidden_state  # (B, Tv, Dv)",
        "print(feats.shape)",
        "```",
        "",
        "## Using the projector (Gemma-style multimodal path)",
        "",
        "The projector here is provided as a **state dict** plus metadata. It is intended for users",
        "who are wiring a Gemma-style VLM, where the projector maps the vision sequence to a fixed number",
        "of image tokens at the LLM hidden size.",
        "",
        "Two common paths:",
        "",
        "1) **Use with Transformers' Gemma 3 model**: load the full VLM, then load this projector's state_dict",
        "   into the model's `multi_modal_projector` module.",
        "",
        "```python",
        "import torch",
        "from transformers import Gemma3ForConditionalGeneration",
        "",
        'repo_id = "<your-username>/<your-repo>"',
        "vlm = Gemma3ForConditionalGeneration.from_pretrained('google/gemma-3-4b-pt', device_map='cpu')",
        "sd = torch.load('projector_state_dict.pt', map_location='cpu')  # or from the repo checkout",
        "vlm.multi_modal_projector.load_state_dict(sd, strict=False)",
        "vlm.eval()",
        "```",
        "",
        "2) **Recreate the projector module from the class name**, instantiate it, and load the state dict.",
        "   The metadata file records the fully qualified class name (FQN).",
        "",
        "```python",
        "import importlib, json, torch",
        "",
        'with open("projector_config.json", "r") as f:',
        "    meta = json.load(f)",
        "fqn = meta.get('projector_fqn')  # e.g., 'transformers.models.gemma3.modeling_gemma3.Gemma3VisionProjector'",
        "mod_name, cls_name = fqn.rsplit('.', 1)",
        "cls = getattr(importlib.import_module(mod_name), cls_name)",
        "projector = cls(**{k: v for k, v in meta.items() if k.endswith('_dim') or k.endswith('_tokens')})",
        "sd = torch.load('projector_state_dict.pt', map_location='cpu')",
        "projector.load_state_dict(sd, strict=False)",
        "projector.eval()",
        "```",
        "",
        "## Shapes (for reference)",
    ]
    if vision_hidden is not None:
        lines.append(f"- Vision hidden size Dv: {vision_hidden}")
    if llm_hidden is not None:
        lines.append(f"- LLM hidden size H: {llm_hidden}")
    if num_image_tokens is not None:
        lines.append(f"- Projector output tokens Ti: {num_image_tokens}")
    if projector_fqn is not None:
        lines.append(f"- Projector class: `{projector_fqn}`")
    lines += [
        "",
        "## License / Terms",
        "",
        "See the `NOTICE` file. Gemma is provided under and subject to the Gemma Terms of Use:",
        "https://ai.google.dev/gemma/terms",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_notice(out_dir: Path):
    text = [
        "Gemma is provided under and subject to the Gemma Terms of Use",
        "found at https://ai.google.dev/gemma/terms",
    ]
    (out_dir / "NOTICE").write_text("\n".join(text), encoding="utf-8")


def _push_to_hub(local_dir: Path, repo_id: str, private: bool):
    if HfApi is None:
        raise RuntimeError("huggingface_hub is not installed. `pip install huggingface_hub` or omit --push-to-hub.")
    api = HfApi()
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    upload_folder(repo_id=repo_id, folder_path=str(local_dir), commit_message="Initial upload: Gemma3 vision + projector")


def _infer_shapes_and_tokens(
    model: Gemma3ForConditionalGeneration,
    vision: SiglipVisionModel,
    image_processor: AutoImageProcessor,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (vision_hidden_dim Dv, llm_hidden_dim H, projector_out_tokens Ti or None)."""
    Dv = getattr(vision.config, "hidden_size", None)
    H = getattr(model.config, "hidden_size", None)

    Ti = None
    proj = getattr(model, "multi_modal_projector", None)
    if proj is None:
        return Dv, H, None

    # Try to derive Ti by a minimal forward pass.
    try:
        img_size = getattr(image_processor, "size", None)
        # Prefer 896x896 if available; otherwise fall back to square size or 896.
        if isinstance(img_size, dict):
            tgt = img_size.get("shortest_edge", None) or img_size.get("height", None) or img_size.get("width", None)
        elif isinstance(img_size, int):
            tgt = img_size
        else:
            tgt = 896
        if tgt is None:
            tgt = 896
        img = Image.new("RGB", (int(tgt), int(tgt)), (127, 127, 127))
        inputs = image_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            vout = vision(pixel_values=inputs["pixel_values"])
            feats = vout.last_hidden_state  # (1, Tv, Dv)
            try:
                pout = proj(feats)  # (1, Ti, H)
            except Exception:
                # Some implementations expect a dict; try common key
                pout = proj({"image_features": feats})
        Ti = int(pout.shape[1])
    except Exception:
        Ti = None  # Could not infer; still fine.

    return Dv, H, Ti


def main():
    parser = argparse.ArgumentParser(description="Extract Gemma 3 vision encoder and projector.")
    parser.add_argument("--model-id", type=str, default="google/gemma-3-4b-pt", help="Source Gemma 3 checkpoint")
    parser.add_argument("--out", type=str, required=True, help="Output directory for the extracted artifacts")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"])
    parser.add_argument("--device-map", type=str, default="cpu", choices=["cpu"], help="Keep cpu to avoid large VRAM use")
    parser.add_argument("--include-projector", action="store_true", default=True, help="Include the Gemma projector weights")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the output folder to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default=None, help="Target repo id like username/repo-name")
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = _dtype_from_str(args.dtype)

    print(f"[1/7] Loading Gemma 3 model: {args.model_id}")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device_map,  # keep on CPU for safety
        low_cpu_mem_usage=True,
    )

    print("[2/7] Locating SigLIP vision encoder inside the VLM...")
    vision = _find_vision_module(model)
    print(f"       Found vision module: {vision.__class__.__name__}")

    print("[3/7] Saving vision encoder (weights + config) ...")
    vision.save_pretrained(out_dir.as_posix())

    print("[4/7] Saving image processor configuration ...")
    # Prefer the image sub-processor if AutoProcessor wraps both text+image
    try:
        proc = AutoProcessor.from_pretrained(args.model_id)
        image_processor = getattr(proc, "image_processor", None)
        if image_processor is None:
            image_processor = AutoImageProcessor.from_pretrained(args.model_id)
    except Exception:
        image_processor = AutoImageProcessor.from_pretrained(args.model_id)
    image_processor.save_pretrained(out_dir.as_posix())

    projector_fqn = None
    Dv = H = Ti = None

    if args.include_projector:
        print("[5/7] Extracting projector weights and metadata ...")
        proj = getattr(model, "multi_modal_projector", None)
        if proj is None:
            print("       WARN: multi_modal_projector not found on the checkpoint; skipping projector export.")
        else:
            # Save state dict
            torch.save(proj.state_dict(), out_dir / "projector_state_dict.pt")
            # Try to infer dims and token count
            Dv, H, Ti = _infer_shapes_and_tokens(model, vision, image_processor)
            projector_fqn = f"{proj.__class__.__module__}.{proj.__class__.__name__}"
            meta = {
                "projector_fqn": projector_fqn,
                "vision_hidden_dim_Dv": int(Dv) if Dv is not None else None,
                "llm_hidden_dim_H": int(H) if H is not None else None,
                "output_tokens_Ti": int(Ti) if Ti is not None else None,
            }
            (out_dir / "projector_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"       Projector class: {projector_fqn}")
            if Dv is not None:
                print(f"       Vision hidden size Dv: {Dv}")
            if H is not None:
                print(f"       LLM hidden size H: {H}")
            if Ti is not None:
                print(f"       Projector output tokens Ti: {Ti}")
            else:
                print("       Could not infer Ti via forward pass; recorded as null in projector_config.json")

    print("[6/7] Writing README and NOTICE ...")
    _safe_make_readme(
        model_id=args.model_id,
        out_dir=out_dir,
        vision_hidden=Dv,
        llm_hidden=H,
        num_image_tokens=Ti,
        projector_fqn=projector_fqn,
    )
    _write_notice(out_dir)

    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("--push-to-hub requires --repo-id (e.g., username/gemma3-vision-extracted)")
        print(f"[7/7] Pushing to Hub as {args.repo_id} (private={args.private}) ...")
        _push_to_hub(out_dir, args.repo_id, private=bool(args.private))
        print("       Push complete.")
    else:
        print("[7/7] Done. Local artifacts written to:", out_dir.as_posix())


if __name__ == "__main__":
    main()
