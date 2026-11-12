"""Fast MRI dataset inspector (optimized for WSL2 /mnt/c mounts)."""

import argparse
import os
from collections import Counter
from pathlib import Path

import nibabel as nib

SUFFIXES = (".nii.gz", ".nii", ".mha")


def sample_mri_files(folder: Path, max_subjects: int = 5) -> tuple[int, list[Path]]:
    """Count total sub-* folders fast without walking all of /mnt/c, and sample a few."""
    files: list[Path] = []
    sub_dirs = sorted(
        p for p in folder.iterdir() if p.is_dir() and p.name.startswith("sub-")
    )
    if not sub_dirs:
        for child in sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")):
            sub_dirs = sorted(p for p in child.iterdir() if p.is_dir() and p.name.startswith("sub-"))
            if sub_dirs:
                break

    search_roots = sub_dirs[:max_subjects] if sub_dirs else [folder]
    for root_dir in search_roots:
        for root, dirs, names in os.walk(root_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "derivatives"]
            for name in names:
                if not name.startswith("._") and name.endswith(SUFFIXES):
                    files.append(Path(root) / name)
            if len(files) >= 20:
                break
    return len(sub_dirs), sorted(files)


def summarize(folder: Path) -> None:
    print(f"Checking {folder} ...", flush=True)
    num_subjects, sampled_files = sample_mri_files(folder)
    if not sampled_files:
        print(f"  -> {num_subjects} sub-* folders, 0 scans found in sample\n", flush=True)
        return

    mods = Counter(f.name.split(".")[0].split("_")[-1] for f in sampled_files)
    shapes: Counter[tuple[int, ...]] = Counter()
    spacings: Counter[tuple[float, ...]] = Counter()

    for path in sampled_files[:8]:
        try:
            if path.name.endswith(".mha"):
                with path.open("rb") as fh:
                    header = {}
                    for _ in range(25):
                        line = fh.readline().decode("ascii", errors="ignore").strip()
                        if "=" in line:
                            k, v = (s.strip() for s in line.split("=", 1))
                            header[k] = v
                        if line.startswith("ElementDataFile"):
                            break
                if "DimSize" in header:
                    shapes[tuple(int(x) for x in header["DimSize"].split())] += 1
                if "ElementSpacing" in header:
                    spacings[tuple(round(float(x), 2) for x in header["ElementSpacing"].split()[:3])] += 1
            else:
                img = nib.load(path)
                shapes[tuple(int(d) for d in img.shape)] += 1
                spacings[tuple(round(float(z), 2) for z in img.header.get_zooms()[:3])] += 1
        except Exception as exc:
            print(f"  [warn] {path.name}: {exc}", flush=True)

    est_scans = f"~{num_subjects * (len(sampled_files) // max(1, min(num_subjects, 5)))}" if num_subjects else str(len(sampled_files))
    print(f"  subjects   : {num_subjects if num_subjects else 'N/A (non-BIDS)'} (est. scans: {est_scans})")
    print(f"  modalities : {', '.join(k for k, _ in mods.most_common(5))}")
    print(f"  shapes     : {', '.join(f'{k}' for k, _ in shapes.most_common(3))}")
    print(f"  spacing mm : {', '.join(f'{k}' for k, _ in spacings.most_common(3))}\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("/data")])
    args = parser.parse_args()

    for root in args.paths:
        subdirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        targets = [root] if any(d.name.startswith("sub-") for d in subdirs) or not subdirs else subdirs
        for target in targets:
            summarize(target)


if __name__ == "__main__":
    main()
