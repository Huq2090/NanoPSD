#!/usr/bin/env python3
"""
NanoPSD - Minimal Entry Point
Delegates argument parsing to scripts/cli.py and runs the pipeline.
"""

import os
from scripts.cli import parse_args
from pipeline.analyzer import NanoparticleAnalyzer

def main() -> None: 
    args = parse_args()

    # Ensure Output directories exist
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # Instantiate analyzer and run the full pipeline
    analyzer = NanoparticleAnalyzer(
        image_path = args.input,
        scale_bar_nm = args.scale, 
        batch = (args.mode == "batch"), 
        min_size_px = args.min_size,
        mode = args.algo, 
    )
    analyzer.run()

if __name__ == "__main__":
    main()

