# Dual-Pathway Memory Reconstruction

Art+ML proof-of-concept comparing two "memory reconstruction" pathways from a single input photo.

- **Pathway A (visual recall)** — iterative image-to-image diffusion. Each iteration feeds the previous output back through Stable Diffusion img2img.
- **Pathway B (narrative recall)** — iterative image→BLIP caption→text-to-image generation. Each iteration captions the previous output, then generates a new image purely from that caption.

Both pathways run N iterations, save all intermediates, and produce labeled grids for comparison.

## Requirements

- Python 3.10+
- CUDA GPU (targeted at Google Colab T4, ~4 GB VRAM)
- Hugging Face account with access to [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

## Setup

```bash
conda create -n artml-memory python=3.10 -y
conda activate artml-memory
pip install -r requirements.txt
huggingface-cli login
```

## Usage

```bash
python compare_pathways.py --input photo.jpg
```

### CLI Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--input` | str | — | Path to input photo (required) |
| `--outdir` | str | `output` | Output directory |
| `--iters` | int | `5` | Number of reconstruction iterations |
| `--prompt` | str | `reconstructed from memory` | Prompt for pathway A img2img |
| `--strength` | float | `0.55` | Denoising strength for img2img |
| `--guidance` | float | `7.5` | Guidance scale |
| `--steps` | int | `30` | Inference steps |
| `--seed` | int | `42` | Base random seed |
| `--model` | str | `runwayml/stable-diffusion-v1-5` | Stable Diffusion model ID |

### Quick test run

```bash
python compare_pathways.py --input photo.jpg --iters 2 --steps 15
```

## Output

```
output/
  pathway_a/
    iter_00.png          # original (resized)
    iter_01.png .. iter_05.png
    grid_a.png           # labeled horizontal strip
  pathway_b/
    iter_00.png          # original (resized)
    iter_01.png .. iter_05.png
    grid_b.png           # labeled horizontal strip
    captions.txt         # BLIP caption per iteration
```

## How It Works

Both pathways start from the same resized input image. At each iteration:

- **Pathway A** passes the current image through SD img2img with a fixed prompt, producing a progressively degraded/reinterpreted version — analogous to recalling a visual memory.
- **Pathway B** captions the current image with BLIP, then generates a new image from that caption alone via SD txt2img — analogous to recalling a memory through its verbal description.

The seed for each iteration is `base_seed + iteration`, keeping randomness consistent across pathways so differences reflect the method, not the noise.

SD img2img and txt2img share weights in memory (~3.4 GB), with BLIP adding ~0.45 GB, fitting comfortably on a Colab T4.
