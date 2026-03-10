#!/usr/bin/env python3
"""
download_weights.py  –  Download pretrained model weights for Chisel.

Downloads:
  • SuperPoint  (MagicLeap, via SuperGlue release)
  • LightGlue   (ETH Zurich cvg lab)  [optional]

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --output weights/ --no-lightglue
"""

import argparse
import sys
import urllib.request
import hashlib
from pathlib import Path


# ---------------------------------------------------------------------------
# Weight sources
# ---------------------------------------------------------------------------

WEIGHTS = {
    "superpoint": {
        "url": (
            "https://github.com/magicleap/SuperGluePretrainedNetwork"
            "/raw/master/models/weights/superpoint_v1.pth"
        ),
        "filename": "superpoint_v1.pth",
        "sha256": None,   # set to hex string to enable verification
        "description": "SuperPoint v1 (MagicLeap)",
    },
    "lightglue_superpoint": {
        "url": (
            "https://github.com/cvg/LightGlue"
            "/releases/download/v0.1_arxiv/superpoint_lightglue.pth"
        ),
        "filename": "superpoint_lightglue.pth",
        "sha256": None,
        "description": "LightGlue trained on SuperPoint descriptors (ETH cvg)",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path):
    print(f"  Downloading {dest.name} …")
    try:
        def _progress(count, block, total):
            done = count * block
            pct  = min(100, int(done * 100 / total)) if total > 0 else 0
            bar  = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%  {done // 1024:,} KB", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
    except Exception as e:
        print(f"\n  [error] Download failed: {e}")
        print(f"  Please download manually from:\n    {url}")
        if dest.exists():
            dest.unlink()
        return False
    return True


def _verify(path: Path, expected: str) -> bool:
    if expected is None:
        return True
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        print(f"  [error] Checksum mismatch for {path.name}")
        print(f"    expected: {expected}")
        print(f"    got:      {actual}")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained weights for Chisel perception models")
    parser.add_argument("--output", default="weights",
                        help="Directory to save weights (default: weights/)")
    parser.add_argument("--no-superpoint", action="store_true",
                        help="Skip SuperPoint weights")
    parser.add_argument("--no-lightglue", action="store_true",
                        help="Skip LightGlue weights")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if file already exists")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nChisel weight downloader")
    print(f"Output directory: {out.resolve()}\n")

    to_download = []
    if not args.no_superpoint:
        to_download.append("superpoint")
    if not args.no_lightglue:
        to_download.append("lightglue_superpoint")

    ok_count = 0
    for key in to_download:
        info = WEIGHTS[key]
        dest = out / info["filename"]

        print(f"{'─'*50}")
        print(f"  {info['description']}")
        print(f"  → {dest}")

        if dest.exists() and not args.force:
            print(f"  Already exists, skipping (use --force to re-download)")
            ok_count += 1
            continue

        if not _download(info["url"], dest):
            continue

        if not _verify(dest, info["sha256"]):
            continue

        size_mb = dest.stat().st_size / (1024 ** 2)
        print(f"  Saved {size_mb:.1f} MB")
        ok_count += 1

    print(f"\n{'─'*50}")
    print(f"  {ok_count}/{len(to_download)} weight files ready")
    print()

    if ok_count > 0:
        print("Usage in pipeline:")
        print("  # SuperPoint with pretrained weights")
        print("  from chisel.perception.feature_extractor import SuperPointExtractor")
        print("  extractor = SuperPointExtractor(")
        print(f"      weights_path='{out}/superpoint_v1.pth')")
        print()
        print("  # LightGlue matcher with pretrained weights")
        print("  from chisel.perception.feature_matcher import LightGlueMatcher")
        print("  matcher = LightGlueMatcher(")
        print(f"      weights_path='{out}/superpoint_lightglue.pth')")
        print()
        print("  # Or pass via PipelineConfig (edit pipeline.py _setup_modules)")

    if ok_count < len(to_download):
        sys.exit(1)


if __name__ == "__main__":
    main()
