# Multi-Pathway Memory Reconstruction

Art+ML proof-of-concept comparing memory reconstruction pathways from a single input photo.

- **Pathway A (visual recall)** — iterative image-to-image diffusion. Each iteration feeds the previous output back through Stable Diffusion img2img.
- **Pathway B (narrative recall)** — iterative image→BLIP caption→text-to-image generation. Each iteration captions the previous output, then generates a new image purely from that caption.
- **Pathway C (dream recall)** — ControlNet Canny edge conditioning + BLIP captions. Spatial layout is preserved via edge maps while semantic content drifts freely. Two variants:
  - **C-fixed**: Canny edges from the *original* image every iteration (structure fully locked)
  - **C-drift**: Canny edges from the *current* iteration's output (structure drifts slowly)
- **Pathway D (false memory)** — same caption→txt2img loop as Pathway B, except at iteration *k* the BLIP caption is replaced with a *misleading* one (e.g. "a dog in the kitchen" when there is no dog). Subsequent iterations resume normal BLIP captioning, so any persistence of the injected detail is a property of the reconstruction process. Implements the Loftus & Palmer (1974) misinformation paradigm in a generative setting.

## Milestones

- **Milestone 1**: Pathway A (visual recall) and Pathway B (narrative recall) — iterative image degradation through visual and linguistic bottlenecks
- **Milestone 2**: Pathway C (dream recall) — ControlNet-based structure-preserving content drift, with fixed and drifting structure variants
- **Milestone 3**: Pathway D (false memory) — single-shot misleading-caption injection at a chosen iteration; persistence quantified by CLIP probe similarity over the full trajectory, with Pathway B as control

## Requirements

- Python 3.10+
- CUDA GPU (targeted at Google Colab T4, ~4 GB VRAM)
- Hugging Face account with access to [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

## Setup (Google Colab — recommended)

1. Go to [colab.google.com](https://colab.google.com) and create a new notebook
2. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU**
3. Accept the Stable Diffusion license at [huggingface.co/runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) (click "Agree" — requires a free HuggingFace account)
4. Run these cells:

```python
# Cell 1 — Clone repo and install dependencies
!git clone https://github.com/YOUR_USERNAME/memory_simulation.git
%cd memory_simulation
!pip install -r requirements.txt
```

```python
# Cell 2 — HuggingFace login (paste your token from huggingface.co/settings/tokens)
from huggingface_hub import login
login()
```

```python
# Cell 3 — Upload your input photo(s)
from google.colab import files
uploaded = files.upload()  # drag and drop your photos
```

```python
# Cell 4 — Run generation (A + B only)
!python compare_pathways.py --input your_photo.jpg --iters 5 --steps 30

# Or with Pathway C enabled
!python compare_pathways.py --input your_photo.jpg --iters 5 --steps 30 --run-pathway-c
```

```python
# Cell 5 — Run analysis
!python analyze_pathways.py --outdir output
```

Results will be in the `output/` folder — browse via Colab's file panel (left sidebar) or download them.

## Setup (Local)

```bash
conda create -n artml-memory python=3.10 -y
conda activate artml-memory
pip install -r requirements.txt
huggingface-cli login
```

## Scripts

### `compare_pathways.py` — Image Generation

Runs pathways on a single input image.

```bash
# Pathways A + B only (Milestone 1)
python compare_pathways.py --input photo.jpg

# Pathways A + B + C (Milestone 2)
python compare_pathways.py --input photo.jpg --run-pathway-c

# Pathways A + B + D (Milestone 3) — false-memory injection
python compare_pathways.py --input photo.jpg \
    --run-pathway-d --inject-at 3 \
    --inject-caption "a dog sitting in the kitchen" \
    --inject-probe "a dog"

# All four pathways at once
python compare_pathways.py --input photo.jpg \
    --run-pathway-c --run-pathway-d \
    --inject-caption "a dog sitting in the kitchen" --inject-probe "a dog"
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
| `--run-pathway-c` | flag | — | Enable Pathway C (dream recall) |
| `--controlnet-model` | str | `lllyasviel/sd-controlnet-canny` | ControlNet model ID |
| `--controlnet-scale` | float | `1.0` | ControlNet conditioning scale |
| `--canny-low` | int | `100` | Canny edge detection low threshold |
| `--canny-high` | int | `200` | Canny edge detection high threshold |
| `--dream-structure` | str | `both` | `fixed`, `drift`, or `both` |
| `--run-pathway-d` | flag | — | Enable Pathway D (false memory injection) |
| `--inject-at` | int | `3` | Iteration at which to inject the false caption (1-indexed) |
| `--inject-caption` | str | — | Misleading caption used at the injection iteration (required with `--run-pathway-d`) |
| `--inject-probe` | str | =`--inject-caption` | Short text probe used to score persistence of the injected concept |

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
| MSE | Pixel-level mean squared error | A, B, C-fixed, C-drift |
| PSNR | Peak signal-to-noise ratio (dB) | A, B, C-fixed, C-drift |
| SSIM | Structural similarity index | A, B, C-fixed, C-drift |
| LPIPS | Learned perceptual distance (AlexNet) | A, B, C-fixed, C-drift |
| CLIP image sim | Cosine similarity of CLIP image embeddings vs original | A, B, C-fixed, C-drift |
| CLIP text-image sim | How well each caption describes its generated image | B, C-fixed, C-drift |
| CLIP text-original sim | How well each caption describes the original image | B, C-fixed, C-drift, D |
| Caption consecutive sim | CLIP text-text similarity between consecutive captions | B, C-fixed, C-drift, D |
| CLIP probe sim | CLIP(inject_probe, image) per iteration — false-memory persistence | All pathways when D is enabled (D = primary, B = control) |

### `run_experiment.py` — Batch Orchestrator

Runs generation + analysis across multiple input images.

```bash
# Multiple specific images
python run_experiment.py --inputs photo1.jpg photo2.jpg --iters 5 --steps 30

# With Pathway C
python run_experiment.py --inputs photo1.jpg photo2.jpg --run-pathway-c

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
| `--run-pathway-c` | flag | — | Enable Pathway C |
| `--controlnet-model` | str | `lllyasviel/sd-controlnet-canny` | ControlNet model ID |
| `--controlnet-scale` | float | `1.0` | ControlNet conditioning scale |
| `--canny-low` | int | `100` | Canny low threshold |
| `--canny-high` | int | `200` | Canny high threshold |
| `--dream-structure` | str | `both` | `fixed`, `drift`, or `both` |
| `--run-pathway-d` | flag | — | Enable Pathway D |
| `--inject-at` | int | `3` | Iteration at which to inject the false caption |
| `--inject-caption` | str | — | Misleading caption (required with `--run-pathway-d`) |
| `--inject-probe` | str | =`--inject-caption` | Probe text for persistence scoring |

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
    pathway_c_fixed/        # only with --run-pathway-c
      iter_00.png .. iter_05.png
      canny_original.png
      canny_01.png .. canny_05.png
      captions.txt
      grid_c_fixed.png
    pathway_c_drift/        # only with --run-pathway-c
      iter_00.png .. iter_05.png
      canny_original.png
      canny_01.png .. canny_05.png
      captions.txt
      grid_c_drift.png
    pathway_d/              # only with --run-pathway-d
      iter_00.png .. iter_05.png
      captions.txt          # injected iteration tagged [INJECTED]
      grid_d.png
    comparison_grid.png     # N-row side-by-side (up to 5 rows)
  analysis/
    metrics.csv                       # all metrics in flat table
    summary.json                      # mean/std per pathway per iteration
    degradation_curves.png            # 2x3 subplot grid
    caption_drift.png                 # caption similarity + text table
    false_memory_persistence.png      # probe sim vs iteration; D vs B control
```

## Quick Test

```bash
# Generate with 2 iterations (fast, A+B only)
python compare_pathways.py --input photo.jpg --iters 2 --steps 15

# Generate with Pathway C
python compare_pathways.py --input photo.jpg --iters 2 --steps 15 --run-pathway-c

# Analyze (CPU fallback if no GPU available for analysis)
python analyze_pathways.py --outdir output --device cpu
```

## Paper Integration

The outputs map to paper figures:

- **Figure 1**: `comparison_grid.png` — visual side-by-side (up to 5 rows with Pathways C and D)
- **Figure 2**: `degradation_curves.png` — quantitative degradation curves (all pathways)
- **Figure 3**: `caption_drift.png` — semantic drift analysis (B, C-fixed, C-drift, D)
- **Figure 4**: `false_memory_persistence.png` — CLIP probe similarity per iteration; D trajectory vs. B control, with the injection iteration marked
- **Table 1**: Pull from `summary.json` — final-iteration metrics averaged across inputs

## How It Works

All pathways start from the same resized input image. At each iteration:

- **Pathway A** passes the current image through SD img2img with a fixed prompt, producing a progressively degraded/reinterpreted version — analogous to recalling a visual memory.
- **Pathway B** captions the current image with BLIP, then generates a new image from that caption alone via SD txt2img — analogous to recalling a memory through its verbal description.
- **Pathway C** captions the current image with BLIP (like B), but generates via ControlNet conditioned on Canny edge maps — analogous to dream recall where spatial structure persists but content drifts. In **C-fixed** mode, edges come from the original image (structure locked); in **C-drift** mode, edges come from the current iteration's output (gradual structural dissolution).
- **Pathway D** is Pathway B with a single perturbation: at iteration `--inject-at`, BLIP is bypassed and the prompt is replaced with `--inject-caption` (a misleading description). All other iterations use BLIP normally. This models eyewitness misinformation: a one-time piece of false information enters the recall chain, and we measure whether the false detail subsequently propagates. Persistence is scored as CLIP similarity between `--inject-probe` (a short text describing the false concept) and each iteration's image; Pathway B run on the same input acts as the no-injection control.

The seed for each iteration is `base_seed + iteration`, keeping randomness consistent across pathways so differences reflect the method, not the noise.

SD img2img and txt2img share weights in memory (~3.4 GB), with ControlNet adding ~1.3 GB and BLIP adding ~0.45 GB (~5.15 GB total with Pathway C), fitting on a Colab T4. Pathway D adds no new model weights — it reuses the txt2img + BLIP pipeline from Pathway B. LPIPS and CLIP models for analysis load separately (~580 MB total).
