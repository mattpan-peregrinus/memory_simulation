"""Quantitative analysis of dual-pathway memory reconstruction outputs.

Computes per-iteration metrics (MSE, PSNR, SSIM, LPIPS, CLIP similarity)
comparing each reconstructed image to the original, and produces CSV/JSON
summaries plus publication-ready plots.

Operates on saved PNGs so generation models need not be loaded concurrently.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Scorer wrappers (instantiated once, reused across all images)
# ---------------------------------------------------------------------------

class LPIPSScorer:
    """Wraps the lpips package for perceptual distance."""

    def __init__(self, device="cpu"):
        import lpips
        self.device = device
        self.model = lpips.LPIPS(net="alex").to(device)

    @staticmethod
    def _to_tensor(img):
        import torch
        arr = np.array(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t * 2 - 1  # scale to [-1, 1]

    def score(self, img_a, img_b):
        import torch
        with torch.no_grad():
            ta = self._to_tensor(img_a).to(self.device)
            tb = self._to_tensor(img_b).to(self.device)
            return self.model(ta, tb).item()


class CLIPScorer:
    """Wraps open_clip for image-image and text-image cosine similarity."""

    def __init__(self, device="cpu"):
        import open_clip
        import torch
        self.device = device
        self.torch = torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def _encode_image(self, img):
        import torch
        with torch.no_grad():
            t = self.preprocess(img).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat

    def _encode_text(self, text):
        import torch
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            feat = self.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat

    def image_similarity(self, img_a, img_b):
        fa, fb = self._encode_image(img_a), self._encode_image(img_b)
        return (fa @ fb.T).item()

    def text_image_similarity(self, text, img):
        ft = self._encode_text(text)
        fi = self._encode_image(img)
        return (ft @ fi.T).item()

    def text_text_similarity(self, text_a, text_b):
        fa = self._encode_text(text_a)
        fb = self._encode_text(text_b)
        return (fa @ fb.T).item()


# ---------------------------------------------------------------------------
# Basic image metrics (no model weights needed)
# ---------------------------------------------------------------------------

def compute_mse(img_a, img_b):
    a = np.array(img_a, dtype=np.float64)
    b = np.array(img_b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def compute_psnr(mse_val, max_val=255.0):
    if mse_val == 0:
        return float("inf")
    return float(10 * math.log10(max_val ** 2 / mse_val))


def compute_ssim(img_a, img_b):
    a = np.array(img_a)
    b = np.array(img_b)
    return float(ssim(a, b, channel_axis=2, data_range=255))


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

ALL_PATHWAYS = [
    ("pathway_a", "A"),
    ("pathway_b", "B"),
    ("pathway_c_fixed", "C-fixed"),
    ("pathway_c_drift", "C-drift"),
    ("pathway_d", "D"),
]


def discover_images(outdir):
    """Return list of image subdirectory names that have at least one pathway dir."""
    outdir = Path(outdir)
    names = []
    for d in sorted(outdir.iterdir()):
        if d.is_dir() and any((d / pw_dir).is_dir() for pw_dir, _ in ALL_PATHWAYS):
            names.append(d.name)
    return names


def load_iteration_images(pathway_dir, iters):
    """Load iter_00.png .. iter_{iters:02d}.png, return list of PIL Images."""
    imgs = []
    for i in range(iters + 1):
        p = pathway_dir / f"iter_{i:02d}.png"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


def load_captions(pathway_dir):
    """Parse captions.txt → dict mapping iteration int → caption string.

    Strips the [INJECTED] marker used by Pathway D so downstream metrics see
    the actual prompt text, not the marker.
    """
    cap_file = pathway_dir / "captions.txt"
    if not cap_file.exists():
        return {}
    captions = {}
    for line in cap_file.read_text().strip().splitlines():
        m = re.match(r"iter_(\d+):\s*(.*)", line)
        if m:
            text = m.group(2)
            if text.startswith("[INJECTED] "):
                text = text[len("[INJECTED] "):]
            captions[int(m.group(1))] = text
    return captions


def load_inject_meta(img_dir):
    """Return (inject_at, inject_probe) from metadata.json, or (None, None)."""
    meta_path = img_dir / "metadata.json"
    if not meta_path.exists():
        return None, None
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None, None
    return meta.get("inject_at"), meta.get("inject_probe")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(args):
    outdir = Path(args.outdir)
    analysis_dir = outdir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    image_names = discover_images(outdir)
    if not image_names:
        print(f"ERROR: No image subdirectories found in {outdir}")
        sys.exit(1)
    print(f"Found {len(image_names)} image(s): {', '.join(image_names)}")

    # Determine iteration count
    iters = args.iters
    if iters is None:
        # Auto-detect from first available pathway
        first_img_dir = outdir / image_names[0]
        first_dir = None
        for pw_dir_name, _ in ALL_PATHWAYS:
            candidate = first_img_dir / pw_dir_name
            if candidate.is_dir():
                first_dir = candidate
                break
        if first_dir is None:
            print("ERROR: No pathway directories found.")
            sys.exit(1)
        iters = len(list(first_dir.glob("iter_*.png"))) - 1
        if iters < 1:
            print("ERROR: Could not auto-detect iteration count.")
            sys.exit(1)
        print(f"Auto-detected {iters} iterations")

    # Load optional scorers
    lpips_scorer = None
    clip_scorer = None
    device = args.device

    if not args.no_lpips:
        print("Loading LPIPS model...")
        lpips_scorer = LPIPSScorer(device=device)

    if not args.no_clip:
        print("Loading CLIP model...")
        clip_scorer = CLIPScorer(device=device)

    # Collect rows
    rows = []

    for img_name in image_names:
        print(f"\nAnalyzing: {img_name}")
        img_dir = outdir / img_name
        inject_at, inject_probe = load_inject_meta(img_dir)

        for pathway, pw_label in ALL_PATHWAYS:
            pw_dir = img_dir / pathway
            if not pw_dir.is_dir():
                continue
            images = load_iteration_images(pw_dir, iters)
            original = images[0]

            has_captions = pathway != "pathway_a"
            captions = {}
            if has_captions:
                captions = load_captions(pw_dir)

            for i, img in enumerate(images):
                row = {
                    "image_name": img_name,
                    "pathway": pw_label,
                    "iteration": i,
                    "mse": compute_mse(original, img),
                }
                row["psnr"] = compute_psnr(row["mse"])
                row["ssim"] = compute_ssim(original, img)

                if lpips_scorer is not None:
                    row["lpips"] = lpips_scorer.score(original, img)
                else:
                    row["lpips"] = None

                if clip_scorer is not None:
                    row["clip_image_sim"] = clip_scorer.image_similarity(original, img)
                else:
                    row["clip_image_sim"] = None

                cap = captions.get(i, "")
                row["caption"] = cap

                if has_captions and cap and clip_scorer is not None:
                    row["clip_text_image_sim"] = clip_scorer.text_image_similarity(cap, img)
                    row["clip_text_orig_sim"] = clip_scorer.text_image_similarity(cap, original)
                else:
                    row["clip_text_image_sim"] = None
                    row["clip_text_orig_sim"] = None

                if has_captions and i >= 2 and clip_scorer is not None:
                    prev_cap = captions.get(i - 1, "")
                    if cap and prev_cap:
                        row["caption_consecutive_sim"] = clip_scorer.text_text_similarity(prev_cap, cap)
                    else:
                        row["caption_consecutive_sim"] = None
                else:
                    row["caption_consecutive_sim"] = None

                # When the experiment includes a false-memory probe, score every
                # pathway's image against the probe — Pathway D shows persistence
                # of the injected concept; the others (B especially) act as
                # controls that never received the injection.
                if inject_probe and clip_scorer is not None:
                    row["clip_probe_sim"] = clip_scorer.text_image_similarity(inject_probe, img)
                    row["inject_at"] = inject_at
                    row["inject_probe"] = inject_probe
                    row["is_injected_iter"] = (pathway == "pathway_d" and i == inject_at)
                else:
                    row["clip_probe_sim"] = None
                    row["inject_at"] = None
                    row["inject_probe"] = None
                    row["is_injected_iter"] = False

                rows.append(row)
                print(f"  {pw_label} iter {i}: MSE={row['mse']:.1f}  SSIM={row['ssim']:.4f}"
                      + (f"  LPIPS={row['lpips']:.4f}" if row['lpips'] is not None else "")
                      + (f"  CLIP={row['clip_image_sim']:.4f}" if row['clip_image_sim'] is not None else "")
                      + (f"  PROBE={row['clip_probe_sim']:.4f}" if row['clip_probe_sim'] is not None else ""))

    # Build DataFrame and save CSV
    df = pd.DataFrame(rows)
    csv_path = analysis_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved: {csv_path}")

    # Summary JSON
    summary = build_summary(df, iters)
    summary_path = analysis_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    print(f"Summary saved: {summary_path}")

    # Plots
    plot_degradation_curves(df, iters, analysis_dir)
    plot_caption_drift(df, iters, analysis_dir)
    plot_false_memory_persistence(df, iters, analysis_dir)

    for img_name in image_names:
        plot_comparison_grid(outdir, img_name, iters)

    print("\nAnalysis complete.")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and math.isinf(obj):
        return "Inf"
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def build_summary(df, iters):
    """Aggregate mean/std per pathway per iteration, plus final-iteration stats."""
    metric_cols = ["mse", "psnr", "ssim", "lpips", "clip_image_sim",
                   "clip_text_image_sim", "clip_text_orig_sim", "caption_consecutive_sim",
                   "clip_probe_sim"]
    summary = {"per_iteration": {}, "final_iteration": {}}

    for pathway in df["pathway"].unique():
        pw_data = df[df["pathway"] == pathway]
        per_iter = {}
        for i in range(iters + 1):
            iter_data = pw_data[pw_data["iteration"] == i]
            stats = {}
            for col in metric_cols:
                if col in iter_data.columns:
                    vals = iter_data[col].dropna()
                    if len(vals) > 0:
                        stats[col] = {"mean": float(vals.mean()), "std": float(vals.std())}
            per_iter[str(i)] = stats
        summary["per_iteration"][pathway] = per_iter

        # Final iteration aggregate
        final = pw_data[pw_data["iteration"] == iters]
        final_stats = {}
        for col in metric_cols:
            if col in final.columns:
                vals = final[col].dropna()
                if len(vals) > 0:
                    final_stats[col] = {"mean": float(vals.mean()), "std": float(vals.std())}
        summary["final_iteration"][pathway] = final_stats

    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_degradation_curves(df, iters, analysis_dir):
    """2x3 subplot grid: SSIM, LPIPS, CLIP image sim / MSE, PSNR, CLIP text-image sim."""
    metrics = [
        ("ssim", "SSIM", True),
        ("lpips", "LPIPS", True),
        ("clip_image_sim", "CLIP Image Similarity", True),
        ("mse", "MSE", True),
        ("psnr", "PSNR (dB)", True),
        ("clip_text_image_sim", "CLIP Text-Image Sim (caption pathways)", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C-fixed": "#2ca02c",
              "C-drift": "#d62728", "D": "#9467bd"}
    all_present = list(df["pathway"].unique())
    iters_range = np.arange(iters + 1)

    for ax, (col, title, both_pathways) in zip(axes, metrics):
        pathways = all_present if both_pathways else [p for p in all_present if p != "A"]
        has_data = False

        for pw in pathways:
            pw_data = df[df["pathway"] == pw]
            if col not in pw_data.columns:
                continue
            means = []
            stds = []
            for i in iters_range:
                vals = pw_data[pw_data["iteration"] == i][col].dropna()
                if len(vals) == 0:
                    means.append(np.nan)
                    stds.append(0)
                else:
                    means.append(vals.mean())
                    stds.append(vals.std())
            means = np.array(means, dtype=float)
            stds = np.array(stds, dtype=float)

            if np.all(np.isnan(means)):
                continue
            has_data = True

            ax.plot(iters_range, means, marker="o", color=colors.get(pw, "#9467bd"),
                    label=f"Pathway {pw}", linewidth=2, markersize=5)
            if np.any(stds > 0):
                ax.fill_between(iters_range, means - stds, means + stds,
                                alpha=0.2, color=colors.get(pw, "#9467bd"))

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(col.upper() if col != "clip_text_image_sim" else "Cosine Sim")
        ax.set_xticks(iters_range)
        if has_data:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    pw_str = " vs ".join(all_present)
    fig.suptitle(f"Degradation Curves: Pathway {pw_str}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = analysis_dir / "degradation_curves.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {save_path}")


def plot_comparison_grid(outdir, img_name, iters):
    """N-row grid for a single input image, one row per available pathway."""
    img_dir = Path(outdir) / img_name

    rows_imgs = []
    row_labels = []
    label_map = {"pathway_a": "Pathway A", "pathway_b": "Pathway B",
                 "pathway_c_fixed": "C-fixed", "pathway_c_drift": "C-drift"}
    for pw_dir_name, pw_label in ALL_PATHWAYS:
        pw_dir = img_dir / pw_dir_name
        if not pw_dir.is_dir():
            continue
        row = []
        for i in range(iters + 1):
            p = pw_dir / f"iter_{i:02d}.png"
            if p.exists():
                row.append(Image.open(p).convert("RGB"))
        if row:
            rows_imgs.append(row)
            row_labels.append(label_map.get(pw_dir_name, pw_label))

    if len(rows_imgs) < 1:
        return

    # All images should be same size; use first as reference
    w, h = rows_imgs[0][0].size
    n = len(rows_imgs[0])
    num_rows = len(rows_imgs)
    label_h = 30
    row_label_w = 80
    padding = 2

    total_w = row_label_w + n * w + (n - 1) * padding
    total_h = label_h + num_rows * h + (num_rows - 1) * padding

    grid = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    # Column headers
    for i in range(n):
        x = row_label_w + i * (w + padding)
        label = "Original" if i == 0 else f"Iter {i}"
        draw.text((x + w // 2 - 20, 4), label, fill=(0, 0, 0), font=small_font)

    # Row labels and images
    for r, (row_imgs, rlabel) in enumerate(zip(rows_imgs, row_labels)):
        y = label_h + r * (h + padding)
        draw.text((4, y + h // 2 - 8), rlabel, fill=(0, 0, 0), font=font)
        for c, img in enumerate(row_imgs):
            x = row_label_w + c * (w + padding)
            grid.paste(img, (x, y))

    save_path = img_dir / "comparison_grid.png"
    grid.save(save_path)
    print(f"Comparison grid saved: {save_path}")


def plot_caption_drift(df, iters, analysis_dir):
    """Plot consecutive caption similarity + caption text table for all caption-producing pathways."""
    caption_pathways = [p for p in df["pathway"].unique() if p != "A"]
    cap_data = df[df["pathway"].isin(caption_pathways)].copy()
    if cap_data.empty:
        return

    has_sim = "caption_consecutive_sim" in cap_data.columns and cap_data["caption_consecutive_sim"].notna().any()
    has_captions = cap_data["caption"].notna().any() and (cap_data["caption"] != "").any()

    if not has_sim and not has_captions:
        return

    n_plots = (1 if has_sim else 0) + (1 if has_captions else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    ax_idx = 0

    pw_colors = {"B": "#ff7f0e", "C-fixed": "#2ca02c", "C-drift": "#d62728", "D": "#9467bd"}

    if has_sim:
        ax = axes[ax_idx]
        ax_idx += 1
        for pw in caption_pathways:
            pw_data = cap_data[cap_data["pathway"] == pw]
            image_names = pw_data["image_name"].unique()
            for img_name in image_names:
                img_data = pw_data[pw_data["image_name"] == img_name]
                iters_vals = img_data["iteration"].values
                sim_vals = img_data["caption_consecutive_sim"].values
                mask = ~pd.isna(sim_vals)
                if mask.any():
                    label = f"{pw} — {img_name}" if len(image_names) > 1 else pw
                    ax.plot(iters_vals[mask], sim_vals[mask].astype(float),
                            marker="o", label=label, color=pw_colors.get(pw, "#9467bd"),
                            linewidth=2, markersize=5)

        pw_str = ", ".join(caption_pathways)
        ax.set_title(f"Consecutive Caption Similarity ({pw_str})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("CLIP Text-Text Cosine Sim")
        ax.set_xticks(range(iters + 1))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    if has_captions:
        ax = axes[ax_idx]
        ax.axis("off")

        # Build caption table
        table_data = []
        for pw in caption_pathways:
            pw_data = cap_data[cap_data["pathway"] == pw]
            for img_name in pw_data["image_name"].unique():
                img_data = pw_data[pw_data["image_name"] == img_name]
                for _, row in img_data.iterrows():
                    if row["caption"]:
                        table_data.append([pw, img_name, int(row["iteration"]), row["caption"]])

        if table_data:
            table = ax.table(
                cellText=table_data,
                colLabels=["Pathway", "Image", "Iteration", "Caption"],
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.auto_set_column_width([0, 1, 2, 3])
            pw_str = ", ".join(caption_pathways)
            ax.set_title(f"Captions per Iteration ({pw_str})", fontsize=12,
                         fontweight="bold", pad=20)

    plt.tight_layout()
    save_path = analysis_dir / "caption_drift.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Caption drift plot saved: {save_path}")


def plot_false_memory_persistence(df, iters, analysis_dir):
    """Plot CLIP(probe, image) per iteration for Pathway D, with B as control.

    Compares whether the injected concept's similarity stays elevated after the
    injection iteration (false memory persists) or decays back toward baseline
    (the system recovers from the lie). Pathway B uses the same probe text but
    never receives the injection, so it serves as the natural control.
    """
    if "clip_probe_sim" not in df.columns:
        return
    probe_data = df[df["clip_probe_sim"].notna()]
    if probe_data.empty or "D" not in probe_data["pathway"].unique():
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    iters_range = np.arange(iters + 1)
    image_names = sorted(probe_data["image_name"].unique())
    inject_iters = probe_data["inject_at"].dropna().unique()

    style = {"D": ("-", "o"), "B": ("--", "s"), "A": (":", "^"),
             "C-fixed": ("--", "x"), "C-drift": (":", "+")}
    color_for_image = ["#9467bd", "#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]

    for idx, img_name in enumerate(image_names):
        img_rows = probe_data[probe_data["image_name"] == img_name]
        probe = img_rows["inject_probe"].dropna().iloc[0]
        base_color = color_for_image[idx % len(color_for_image)]
        for pw in ["D", "B"]:
            pw_rows = img_rows[img_rows["pathway"] == pw].sort_values("iteration")
            if pw_rows.empty:
                continue
            ls, marker = style.get(pw, ("-", "o"))
            label = (f"{pw} — {img_name}" if len(image_names) > 1 else f"Pathway {pw}")
            label += f" (probe: \"{probe}\")" if pw == "D" else " [control]"
            ax.plot(pw_rows["iteration"], pw_rows["clip_probe_sim"].astype(float),
                    linestyle=ls, marker=marker, linewidth=2, markersize=6,
                    color=base_color, alpha=1.0 if pw == "D" else 0.55,
                    label=label)

    if len(inject_iters) == 1:
        ax.axvline(float(inject_iters[0]), linestyle="--", color="black",
                   alpha=0.6, label=f"injection @ iter {int(inject_iters[0])}")
    else:
        for ij in inject_iters:
            ax.axvline(float(ij), linestyle="--", color="black", alpha=0.4)

    ax.set_title("False Memory Persistence — CLIP(probe, image) over iterations",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("CLIP text-image similarity (probe vs. iteration image)")
    ax.set_xticks(iters_range)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = analysis_dir / "false_memory_persistence.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"False memory persistence plot saved: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyze dual-pathway memory reconstruction outputs")
    p.add_argument("--outdir", default="output", help="Root output directory (default: output)")
    p.add_argument("--iters", type=int, default=None,
                   help="Number of iterations (auto-detected if omitted)")
    p.add_argument("--device", default="cuda",
                   help="Torch device for LPIPS/CLIP (default: cuda)")
    p.add_argument("--no-lpips", action="store_true",
                   help="Skip LPIPS computation")
    p.add_argument("--no-clip", action="store_true",
                   help="Skip all CLIP-based metrics")
    return p.parse_args()


# need these for comparison_grid
from PIL import ImageDraw, ImageFont


if __name__ == "__main__":
    args = parse_args()
    analyze(args)
