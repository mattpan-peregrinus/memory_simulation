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

## Scripts

### `compare_pathways.py` — Image Generation

Runs both pathways on a single input image.

```bash
python compare_pathways.py --input photo.jpg
```

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

### `analyze_pathways.py` — Quantitative Analysis

Computes metrics on saved PNGs (no generation models needed). Runs separately so LPIPS/CLIP don't compete for VRAM with SD/BLIP.

```bash
python analyze_pathways.py --outdir output
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--outdir` | str | `output` | Root output directory |
| `--iters` | int | auto-detect | Number of iterations |
| `--device` | str | `cuda` | Torch device for LPIPS/CLIP |
| `--no-lpips` | flag | — | Skip LPIPS computation |
| `--no-clip` | flag | — | Skip all CLIP-based metrics |

**Metrics computed:**

| Metric | Description | Pathways |
|--------|-------------|----------|
| MSE | Pixel-level mean squared error | A, B |
| PSNR | Peak signal-to-noise ratio (dB) | A, B |
| SSIM | Structural similarity index | A, B |
| LPIPS | Learned perceptual distance (AlexNet) | A, B |
| CLIP image sim | Cosine similarity of CLIP image embeddings vs original | A, B |
| CLIP text-image sim | How well each caption describes its generated image | B |
| CLIP text-original sim | How well each caption describes the original image | B |
| Caption consecutive sim | CLIP text-text similarity between consecutive captions | B |

### `run_experiment.py` — Batch Orchestrator

Runs generation + analysis across multiple input images.

```bash
# Multiple specific images
python run_experiment.py --inputs photo1.jpg photo2.jpg --iters 5 --steps 30

# All images in a directory
python run_experiment.py --input-dir ./photos/ --iters 5

# Re-run analysis only (skip generation)
python run_experiment.py --inputs photo1.jpg --skip-generation
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--inputs` | str[] | — | Input image paths |
| `--input-dir` | str | — | Directory of input images |
| `--outdir` | str | `output` | Output directory |
| `--iters` | int | `5` | Number of iterations |
| `--steps` | int | `30` | Inference steps |
| `--strength` | float | `0.55` | Denoising strength |
| `--guidance` | float | `7.5` | Guidance scale |
| `--seed` | int | `42` | Base random seed |
| `--skip-generation` | flag | — | Only run analysis |
| `--no-lpips` | flag | — | Skip LPIPS in analysis |
| `--no-clip` | flag | — | Skip CLIP in analysis |
| `--analysis-device` | str | `cuda` | Device for analysis models |

## Output Structure

```
output/
  <image_name>/
    metadata.json           # generation parameters
    pathway_a/
      iter_00.png .. iter_05.png
      grid_a.png
    pathway_b/
      iter_00.png .. iter_05.png
      grid_b.png
      captions.txt
    comparison_grid.png     # 2-row side-by-side for paper figures
  analysis/
    metrics.csv             # all metrics in flat table
    summary.json            # mean/std per pathway per iteration
    degradation_curves.png  # 2x3 subplot grid
    caption_drift.png       # caption similarity + text table
```

## Quick Test

```bash
# Generate with 2 iterations (fast)
python compare_pathways.py --input photo.jpg --iters 2 --steps 15

# Analyze (CPU fallback if no GPU available for analysis)
python analyze_pathways.py --outdir output --device cpu
```

## Paper Integration

The outputs map to paper figures:

- **Figure 1**: `comparison_grid.png` — visual side-by-side
- **Figure 2**: `degradation_curves.png` — quantitative degradation curves
- **Figure 3**: `caption_drift.png` — semantic drift analysis
- **Table 1**: Pull from `summary.json` — final-iteration metrics averaged across inputs

## How It Works

Both pathways start from the same resized input image. At each iteration:

- **Pathway A** passes the current image through SD img2img with a fixed prompt, producing a progressively degraded/reinterpreted version — analogous to recalling a visual memory.
- **Pathway B** captions the current image with BLIP, then generates a new image from that caption alone via SD txt2img — analogous to recalling a memory through its verbal description.

The seed for each iteration is `base_seed + iteration`, keeping randomness consistent across pathways so differences reflect the method, not the noise.

SD img2img and txt2img share weights in memory (~3.4 GB), with BLIP adding ~0.45 GB, fitting comfortably on a Colab T4. LPIPS and CLIP models for analysis load separately (~580 MB total).
