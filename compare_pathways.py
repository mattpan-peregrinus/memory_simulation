"""Multi-Pathway Memory Reconstruction.

Compares memory reconstruction pathways from a single input photo:
  Pathway A (visual recall)     — iterative img2img diffusion
  Pathway B (narrative recall)  — iterative image→caption→txt2img generation
  Pathway C (dream recall)      — ControlNet Canny edge conditioning + BLIP captions
  Pathway D (false memory)      — Pathway B with one misleading caption injected
                                  at iteration k (Loftus-style memory implantation)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
)
from transformers import BlipProcessor, BlipForConditionalGeneration


def parse_args():
    p = argparse.ArgumentParser(description="Multi-pathway memory reconstruction")
    p.add_argument("--input", required=True, help="Path to input photo")
    p.add_argument("--outdir", default="output", help="Output directory")
    p.add_argument("--iters", type=int, default=5, help="Number of reconstruction iterations")
    p.add_argument("--prompt", default="reconstructed from memory", help="Prompt for pathway A img2img")
    p.add_argument("--strength", type=float, default=0.55, help="Denoising strength for img2img")
    p.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    p.add_argument("--steps", type=int, default=30, help="Inference steps")
    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model ID")
    # Pathway C (dream recall) args
    p.add_argument("--run-pathway-c", action="store_true", help="Enable Pathway C (dream recall via ControlNet)")
    p.add_argument("--controlnet-model", default="lllyasviel/sd-controlnet-canny",
                   help="ControlNet model ID")
    p.add_argument("--controlnet-scale", type=float, default=1.0,
                   help="ControlNet conditioning scale")
    p.add_argument("--canny-low", type=int, default=100, help="Canny edge low threshold")
    p.add_argument("--canny-high", type=int, default=200, help="Canny edge high threshold")
    p.add_argument("--dream-structure", choices=["fixed", "drift", "both"], default="both",
                   help="C-fixed (edges from original), C-drift (edges from current), or both")
    # Pathway D (false memory) args
    p.add_argument("--run-pathway-d", action="store_true",
                   help="Enable Pathway D (false memory injection)")
    p.add_argument("--inject-at", type=int, default=3,
                   help="Iteration at which to inject the false caption (1-indexed)")
    p.add_argument("--inject-caption", default=None,
                   help="False caption used at the injection iteration in place of BLIP's caption")
    p.add_argument("--inject-probe", default=None,
                   help="Short text probe used to score persistence of the injected concept "
                        "(defaults to --inject-caption)")
    args = p.parse_args()
    if args.run_pathway_d:
        if not args.inject_caption:
            print("ERROR: --run-pathway-d requires --inject-caption")
            sys.exit(1)
        if not (1 <= args.inject_at <= args.iters):
            print(f"ERROR: --inject-at must be in [1, {args.iters}]")
            sys.exit(1)
        if args.inject_probe is None:
            args.inject_probe = args.inject_caption
    return args


def authenticate_hf():
    import os
    try:
        from huggingface_hub import get_token
        token = get_token()
    except ImportError:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()

    if not token:
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        )

    if not token:
        print("ERROR: No Hugging Face token found.")
        print("Set HF_TOKEN or login first.")
        sys.exit(1)


def resize_image(img, max_side=768):
    """Scale to max_side px, round dimensions to multiple of 8."""
    w, h = img.size
    scale = min(max_side / w, max_side / h, 1.0)
    w, h = int(w * scale), int(h * scale)
    w, h = w - w % 8, h - h % 8
    return img.resize((w, h), Image.LANCZOS)


def extract_canny_edges(pil_img, low, high):
    """Convert PIL Image to Canny edge map as a 3-channel RGB PIL Image."""
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_rgb)


def load_models(model_id, device, load_controlnet=False, controlnet_model_id=None):
    """Load img2img pipe, derive txt2img from shared components, load BLIP.

    When load_controlnet=True, also load ControlNet and construct a
    StableDiffusionControlNetPipeline sharing SD weights.
    """
    print("Loading Stable Diffusion img2img pipeline...")
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    print("Constructing txt2img pipeline (shared weights)...")
    pipe_txt2img = StableDiffusionPipeline(
        vae=pipe_img2img.vae,
        text_encoder=pipe_img2img.text_encoder,
        tokenizer=pipe_img2img.tokenizer,
        unet=pipe_img2img.unet,
        scheduler=pipe_img2img.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)

    pipe_controlnet = None
    if load_controlnet:
        print(f"Loading ControlNet: {controlnet_model_id}...")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id, torch_dtype=torch.float16
        ).to(device)
        print("Constructing ControlNet pipeline (shared SD weights)...")
        pipe_controlnet = StableDiffusionControlNetPipeline(
            vae=pipe_img2img.vae,
            text_encoder=pipe_img2img.text_encoder,
            tokenizer=pipe_img2img.tokenizer,
            unet=pipe_img2img.unet,
            scheduler=pipe_img2img.scheduler,
            controlnet=controlnet,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(device)

    print("Loading BLIP captioning model...")
    blip_id = "Salesforce/blip-image-captioning-base"
    blip_processor = BlipProcessor.from_pretrained(blip_id)
    blip_model = BlipForConditionalGeneration.from_pretrained(
        blip_id, torch_dtype=torch.float16
    ).to(device)

    return pipe_img2img, pipe_txt2img, pipe_controlnet, blip_processor, blip_model


def caption_image(img, blip_processor, blip_model, device):
    """Generate a BLIP caption for an image."""
    inputs = blip_processor(img, return_tensors="pt").to(device, torch.float16)
    ids = blip_model.generate(**inputs, max_new_tokens=50)
    return blip_processor.decode(ids[0], skip_special_tokens=True)


def run_pathway_a(img, pipe_img2img, args, out_dir, device):
    """Pathway A: visual recall via iterative img2img."""
    out_dir.mkdir(parents=True, exist_ok=True)
    current = img.copy()
    current.save(out_dir / "iter_00.png")
    images = [current]

    for i in range(1, args.iters + 1):
        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        result = pipe_img2img(
            prompt=args.prompt,
            image=current,
            strength=args.strength,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            generator=gen,
        ).images[0]
        result.save(out_dir / f"iter_{i:02d}.png")
        print(f"  Pathway A  iter {i}/{args.iters}")
        images.append(result)
        current = result

    return images


def run_pathway_b(img, pipe_txt2img, blip_processor, blip_model, args, out_dir, device):
    """Pathway B: narrative recall via caption→txt2img loop."""
    out_dir.mkdir(parents=True, exist_ok=True)
    current = img.copy()
    current.save(out_dir / "iter_00.png")
    images = [current]
    captions = []

    for i in range(1, args.iters + 1):
        cap = caption_image(current, blip_processor, blip_model, device)
        captions.append(f"iter_{i:02d}: {cap}")
        print(f"  Pathway B  iter {i}/{args.iters}  caption: {cap}")

        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        result = pipe_txt2img(
            prompt=cap,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            height=current.size[1],
            width=current.size[0],
            generator=gen,
        ).images[0]
        result.save(out_dir / f"iter_{i:02d}.png")
        images.append(result)
        current = result

    (out_dir / "captions.txt").write_text("\n".join(captions) + "\n")
    return images


def run_pathway_c(img, pipe_controlnet, blip_processor, blip_model, args, out_dir, device, structure_mode):
    """Pathway C: dream recall via ControlNet Canny edges + BLIP captions.

    structure_mode: 'fixed' — Canny from original every iter
                    'drift' — Canny from current iter's output
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    current = img.copy()
    current.save(out_dir / "iter_00.png")
    images = [current]
    captions = []

    # Pre-compute original Canny edges
    original_canny = extract_canny_edges(img, args.canny_low, args.canny_high)
    original_canny.save(out_dir / "canny_original.png")

    for i in range(1, args.iters + 1):
        cap = caption_image(current, blip_processor, blip_model, device)
        captions.append(f"iter_{i:02d}: {cap}")
        print(f"  Pathway C-{structure_mode}  iter {i}/{args.iters}  caption: {cap}")

        if structure_mode == "fixed":
            canny_map = original_canny
        else:
            canny_map = extract_canny_edges(current, args.canny_low, args.canny_high)

        canny_map.save(out_dir / f"canny_{i:02d}.png")

        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        result = pipe_controlnet(
            prompt=cap,
            image=canny_map,
            controlnet_conditioning_scale=args.controlnet_scale,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            height=current.size[1],
            width=current.size[0],
            generator=gen,
        ).images[0]
        result.save(out_dir / f"iter_{i:02d}.png")
        images.append(result)
        current = result

    (out_dir / "captions.txt").write_text("\n".join(captions) + "\n")
    return images


def run_pathway_d(img, pipe_txt2img, blip_processor, blip_model, args, out_dir, device):
    """Pathway D: false memory injection.

    Identical to Pathway B (caption→txt2img loop) except that at iteration
    `args.inject_at` the BLIP caption is replaced with `args.inject_caption`.
    Subsequent iterations resume normal BLIP captioning of the post-injection
    image, so any persistence of the injected detail is a property of the
    reconstruction process, not of repeated prompting.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    current = img.copy()
    current.save(out_dir / "iter_00.png")
    images = [current]
    captions = []

    for i in range(1, args.iters + 1):
        if i == args.inject_at:
            cap = args.inject_caption
            captions.append(f"iter_{i:02d}: [INJECTED] {cap}")
            print(f"  Pathway D  iter {i}/{args.iters}  [INJECTED]: {cap}")
        else:
            cap = caption_image(current, blip_processor, blip_model, device)
            captions.append(f"iter_{i:02d}: {cap}")
            print(f"  Pathway D  iter {i}/{args.iters}  caption: {cap}")

        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        result = pipe_txt2img(
            prompt=cap,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            height=current.size[1],
            width=current.size[0],
            generator=gen,
        ).images[0]
        result.save(out_dir / f"iter_{i:02d}.png")
        images.append(result)
        current = result

    (out_dir / "captions.txt").write_text("\n".join(captions) + "\n")
    return images


def make_grid(images, labels, save_path):
    """Horizontal concatenation of images with labels above each."""
    label_h = 24
    widths = [im.size[0] for im in images]
    heights = [im.size[1] for im in images]
    total_w = sum(widths)
    max_h = max(heights) + label_h

    grid = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    x = 0
    for im, label in zip(images, labels):
        draw.text((x + 4, 2), label, fill=(0, 0, 0), font=font)
        grid.paste(im, (x, label_h))
        x += im.size[0]

    grid.save(save_path)
    print(f"  Grid saved: {save_path}")


def main():
    args = parse_args()
    authenticate_hf()

    device = torch.device("cuda")
    img = Image.open(args.input).convert("RGB")
    img = resize_image(img)
    print(f"Input resized to {img.size[0]}x{img.size[1]}")

    pipe_img2img, pipe_txt2img, pipe_controlnet, blip_proc, blip_model = load_models(
        args.model, device,
        load_controlnet=args.run_pathway_c,
        controlnet_model_id=args.controlnet_model,
    )

    out = Path(args.outdir)
    image_name = Path(args.input).stem
    image_dir = out / image_name
    dir_a = image_dir / "pathway_a"
    dir_b = image_dir / "pathway_b"

    print("Running Pathway A (visual recall)...")
    imgs_a = run_pathway_a(img, pipe_img2img, args, dir_a, device)

    print("Running Pathway B (narrative recall)...")
    imgs_b = run_pathway_b(img, pipe_txt2img, blip_proc, blip_model, args, dir_b, device)

    labels = [f"iter {i}" for i in range(args.iters + 1)]
    make_grid(imgs_a, labels, dir_a / "grid_a.png")
    make_grid(imgs_b, labels, dir_b / "grid_b.png")

    # Pathway C variants
    if args.run_pathway_c:
        c_modes = []
        if args.dream_structure in ("fixed", "both"):
            c_modes.append("fixed")
        if args.dream_structure in ("drift", "both"):
            c_modes.append("drift")

        for mode in c_modes:
            dir_c = image_dir / f"pathway_c_{mode}"
            print(f"Running Pathway C-{mode} (dream recall)...")
            imgs_c = run_pathway_c(
                img, pipe_controlnet, blip_proc, blip_model, args, dir_c, device, mode
            )
            make_grid(imgs_c, labels, dir_c / f"grid_c_{mode}.png")

    if args.run_pathway_d:
        dir_d = image_dir / "pathway_d"
        print(f"Running Pathway D (false memory, inject at iter {args.inject_at})...")
        imgs_d = run_pathway_d(img, pipe_txt2img, blip_proc, blip_model, args, dir_d, device)
        make_grid(imgs_d, labels, dir_d / "grid_d.png")

    metadata = {
        "input": str(Path(args.input).resolve()),
        "image_name": image_name,
        "width": img.size[0],
        "height": img.size[1],
        "iters": args.iters,
        "prompt": args.prompt,
        "strength": args.strength,
        "guidance": args.guidance,
        "steps": args.steps,
        "seed": args.seed,
        "model": args.model,
    }
    if args.run_pathway_c:
        metadata.update({
            "controlnet_model": args.controlnet_model,
            "controlnet_scale": args.controlnet_scale,
            "canny_low": args.canny_low,
            "canny_high": args.canny_high,
            "dream_structure": args.dream_structure,
        })
    if args.run_pathway_d:
        metadata.update({
            "inject_at": args.inject_at,
            "inject_caption": args.inject_caption,
            "inject_probe": args.inject_probe,
        })
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"  Metadata saved: {image_dir / 'metadata.json'}")

    print("Done.")


if __name__ == "__main__":
    main()
