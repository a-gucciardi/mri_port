"""Inspect available MRI datasets, scan counts, shapes, and NIfTI header metadata."""

import argparse
import os
from collections import Counter
from pathlib import Path

import nibabel as nib

NIFTI_SUFFIXES = (".nii.gz", ".nii")
OTHER_MRI_SUFFIXES = (".mha", ".mhd", ".mgz", ".dcm")
ALL_MRI_SUFFIXES = NIFTI_SUFFIXES + OTHER_MRI_SUFFIXES


def find_mri_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and not path.name.startswith("._")
        and path.name.endswith(ALL_MRI_SUFFIXES)
    )


def modality_tag(path: Path) -> str:
    name = path.name
    for suffix in ALL_MRI_SUFFIXES:
        if name.endswith(suffix):
            name = name.removesuffix(suffix)
            break
    return name.split("_")[-1] if "_" in name else name


def discover_datasets(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    subdirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    direct_scans = [
        p for p in root.iterdir() if p.is_file() and p.name.endswith(ALL_MRI_SUFFIXES)
    ]
    if not subdirs or direct_scans or any(d.name.startswith("sub-") for d in subdirs):
        return [root]
    return subdirs


def summarize_dataset(folder: Path, max_headers: int) -> None:
    files = find_mri_files(folder)
    archives = sorted(
        p.name
        for p in folder.rglob("*")
        if p.is_file() and p.suffix in (".zip", ".tar", ".tgz")
    )
    if not files:
        extra = f" (archives: {', '.join(archives[:5])})" if archives else ""
        print(f"- {folder.name} ({folder}): 0 scans{extra}")
        return

    subjects = {
        part
        for path in files
        for part in path.relative_to(folder).parts
        if part.startswith("sub-")
    }
    modalities = Counter(modality_tag(path) for path in files)
    nifti_files = [p for p in files if p.name.endswith(NIFTI_SUFFIXES)]
    sampled = nifti_files if max_headers <= 0 else nifti_files[:max_headers]

    shapes: Counter[tuple[int, ...]] = Counter()
    spacings: Counter[tuple[float, ...]] = Counter()
    dtypes: Counter[str] = Counter()
    orientations: Counter[str] = Counter()

    for path in sampled:
        image = nib.load(path)
        shapes[tuple(int(d) for d in image.shape)] += 1
        zooms = image.header.get_zooms()[:3]
        spacings[tuple(round(float(z), 2) for z in zooms)] += 1
        dtypes[str(image.get_data_dtype())] += 1
        orientations["".join(nib.aff2axcodes(image.affine))] += 1

    subj_str = f"{len(subjects)} sub-* subjects" if subjects else "no sub-* folders"
    mod_str = ", ".join(f"{k}:{v}" for k, v in modalities.most_common(6))
    print(f"- {folder.name} ({folder}):")
    print(f"    scans      : {len(files)} ({subj_str})")
    print(f"    modalities : {mod_str}")
    if sampled:
        shape_str = ", ".join(f"{k} (x{v})" for k, v in shapes.most_common(3))
        zoom_str = ", ".join(f"{k}mm" for k, v in spacings.most_common(3))
        orient_str = ", ".join(f"{k}" for k, v in orientations.most_common(3))
        dtype_str = ", ".join(dtypes.keys())
        print(f"    shapes     : {shape_str} [sampled {len(sampled)}/{len(nifti_files)} NIfTI]")
        print(f"    spacing    : {zoom_str} | orient: {orient_str} | dtype: {dtype_str}")
    if archives:
        print(f"    archives   : {', '.join(archives[:5])}")


def main() -> None:
    default_root = os.environ.get("MRI_DATA_ROOT", "/data")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path(default_root)],
        help="Root data directory or dataset directories (default: $MRI_DATA_ROOT or /data)",
    )
    parser.add_argument(
        "--max-headers",
        type=int,
        default=50,
        help="Max NIfTI headers to inspect per dataset (0 for all, default: 50)",
    )
    args = parser.parse_args()

    for root in args.paths:
        print(f"=== {root} ===")
        for dataset_dir in discover_datasets(root):
            summarize_dataset(dataset_dir, args.max_headers)


if __name__ == "__main__":
    main()
