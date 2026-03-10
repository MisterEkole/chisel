#!/usr/bin/env python3
"""
download_eth3d.py  –  Download ETH3D multi-view evaluation dataset.

Usage:
    python scripts/download_eth3d.py --output /data/eth3d
    python scripts/download_eth3d.py --output /data/eth3d --scenes courtyard delivery_area
"""

import argparse
import sys
from pathlib import Path



from chisel.data.eth3d_dataset import ETH3DDataset, ETH3D_TRAINING_SCENES


def main():
    parser = argparse.ArgumentParser(description="Download ETH3D dataset")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Specific scenes to download (default: all training)")
    parser.add_argument("--split", type=str, default="training",
                        choices=["training", "test"])
    args = parser.parse_args()

    dataset = ETH3DDataset(args.output, split=args.split)

    scenes = args.scenes or ETH3D_TRAINING_SCENES
    print(f"Downloading {len(scenes)} ETH3D scenes to {args.output}...")
    print(f"Scenes: {', '.join(scenes)}\n")

    dataset.download(scenes)
    print("\nDone! Available scenes:", dataset.list_scenes())


if __name__ == "__main__":
    main()
