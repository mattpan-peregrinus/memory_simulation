"""Batch orchestrator for dual-pathway memory reconstruction experiments.

Runs compare_pathways.py on each input image (sequentially, to fit in VRAM),
then runs analyze_pathways.py on the combined output directory.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run multi-image memory reconstruction experiment")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--inputs", nargs="+", help="Input image paths")
    g.add_argument("--input-dir", help="Directory of input images (jpg/png)")
    p.add_argument("--outdir", default="output", help="Output directory (default: output)")
    p.add_argument("--iters", type=int, default=5, help="Number of iterations (default: 5)")
    p.add_argument("--steps", type=int, default=30, help="Inference steps (default: 30)")
    p.add_argument("--strength", type=float, default=0.55, help="Denoising strength (default: 0.55)")
    p.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (default: 7.5)")
    p.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="SD model ID")
    p.add_argument("--skip-generation", action="store_true",
                   help="Skip image generation, only run analysis")
    p.add_argument("--no-lpips", action="store_true", help="Skip LPIPS in analysis")
    p.add_argument("--no-clip", action="store_true", help="Skip CLIP in analysis")
    p.add_argument("--analysis-device", default="cuda", help="Device for analysis models")
    return p.parse_args()


def collect_inputs(args):
    if args.inputs:
        paths = [Path(p) for p in args.inputs]
    else:
        input_dir = Path(args.input_dir)
        paths = sorted(
            p for p in input_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    for p in paths:
        if not p.exists():
            print(f"ERROR: Input not found: {p}")
            sys.exit(1)
    return paths


def run_generation(input_path, args):
    cmd = [
        sys.executable, "compare_pathways.py",
        "--input", str(input_path),
        "--outdir", args.outdir,
        "--iters", str(args.iters),
        "--steps", str(args.steps),
        "--strength", str(args.strength),
        "--guidance", str(args.guidance),
        "--seed", str(args.seed),
        "--model", args.model,
    ]
    print(f"\n{'='*60}")
    print(f"Generating: {input_path.name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: Generation failed for {input_path.name}")
        sys.exit(1)


def run_analysis(args):
    cmd = [
        sys.executable, "analyze_pathways.py",
        "--outdir", args.outdir,
        "--device", args.analysis_device,
    ]
    if args.iters:
        cmd += ["--iters", str(args.iters)]
    if args.no_lpips:
        cmd.append("--no-lpips")
    if args.no_clip:
        cmd.append("--no-clip")
    print(f"\n{'='*60}")
    print("Running analysis...")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Analysis failed.")
        sys.exit(1)


def main():
    args = parse_args()
    inputs = collect_inputs(args)
    print(f"Experiment: {len(inputs)} image(s), {args.iters} iterations")

    if not args.skip_generation:
        for input_path in inputs:
            run_generation(input_path, args)
    else:
        print("Skipping generation (--skip-generation)")

    run_analysis(args)
    print(f"\nExperiment complete. Results in {args.outdir}/analysis/")


if __name__ == "__main__":
    main()
