"""Freeze BOBs T1/T2 or prepared IXI T1 scans and subject-level splits."""

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np

Split = Literal["train", "val", "test"]
DATASET_MODALITIES = {"BOBs": ["T1w", "T2w"], "IXI": ["T1w"]}


@dataclass(frozen=True)
class Scan:
    subject_id: str
    session_id: str
    t1: str
    t2: str | None
    t1_sha256: str
    t2_sha256: str | None
    shape: tuple[int, int, int]
    split: Split


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_scan(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Scan path must stay within data root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_pair(t1: Path, t2: Path | None = None) -> np.ndarray:
    paths = [t1] if t2 is None else [t1, t2]
    images = [nib.as_closest_canonical(nib.load(path)) for path in paths]
    if any(len(image.shape) != 3 or image.shape != images[0].shape for image in images):
        raise ValueError(f"Expected matching 3D MRI shapes: {paths}")
    if any(
        not np.allclose(images[0].affine, image.affine, rtol=0, atol=1e-4)
        for image in images[1:]
    ):
        raise ValueError(f"T1/T2 are not on the same physical grid: {paths}")
    pair = np.stack([image.get_fdata(dtype=np.float32) for image in images])
    if not np.isfinite(pair).all():
        raise ValueError(f"Non-finite MRI intensities: {t1}, {t2}")
    if any(np.count_nonzero(channel > 0) < 100 for channel in pair):
        raise ValueError(f"Insufficient positive foreground: {t1}, {t2}")
    return pair


def freeze(root: Path, output: Path, seed: int = 42, dataset: str = "BOBs") -> list[Scan]:
    if dataset not in DATASET_MODALITIES:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if output.exists():
        raise FileExistsError(f"Frozen manifest exists; use a new output path: {output}")
    pattern = "*_T1w.nii.gz" if dataset == "BOBs" else "*_T1w_brain_affine_mni.nii.gz"
    t1_paths = sorted(root.glob(f"sub-*/ses-*/anat/{pattern}"))
    if not t1_paths:
        raise ValueError(f"No {dataset} T1w scans found under {root} matching {pattern}")
    pairs: list[tuple[Path, Path | None]]
    if dataset == "BOBs":
        pairs = [
            (t1, t1.with_name(t1.name.replace("_T1w.nii.gz", "_T2w.nii.gz")))
            for t1 in t1_paths
        ]
        t2_paths = set(root.glob("sub-*/ses-*/anat/*_T2w.nii.gz"))
        expected_t2 = {t2 for _, t2 in pairs if t2 is not None}
        if expected_t2 != t2_paths:
            raise ValueError(
                f"Unpaired scans: missing T2={sorted(expected_t2 - t2_paths)}, "
                f"extra T2={sorted(t2_paths - expected_t2)}"
            )
    else:
        pairs = [(t1, None) for t1 in t1_paths]
    subjects = sorted({path.relative_to(root).parts[0] for path in t1_paths})
    if len(subjects) < 10:
        raise ValueError("At least 10 subjects required for an 80/10/10 split")
    random.Random(seed).shuffle(subjects)
    held_out = max(1, round(len(subjects) * 0.1))
    assignments: dict[str, Split] = {
        subject: "test" if i < held_out else "val" if i < 2 * held_out else "train"
        for i, subject in enumerate(subjects)
    }
    scans = []
    for t1, t2 in pairs:
        relative = t1.relative_to(root)
        if dataset == "IXI":
            expected_name = f"m{relative.parts[0]}_{relative.parts[1]}_T1w_brain_affine_mni.nii.gz"
            if t1.name != expected_name:
                raise ValueError(f"IXI filename does not match subject/session folders: {relative}")
        pair = load_pair(t1, t2)
        scan = Scan(
            subject_id=relative.parts[0],
            session_id=relative.parts[1],
            t1=relative.as_posix(),
            t2=t2.relative_to(root).as_posix() if t2 is not None else None,
            t1_sha256=sha256(t1),
            t2_sha256=sha256(t2) if t2 is not None else None,
            shape=(pair.shape[1], pair.shape[2], pair.shape[3]),
            split=assignments[relative.parts[0]],
        )
        scans.append(scan)
        print(f"[QC] {scan.subject_id}/{scan.session_id}: {scan.shape} -> {scan.split}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "cohort_status": "health_unverified",
        "modalities": DATASET_MODALITIES[dataset],
        "split_seed": seed,
        "scans": [asdict(scan) for scan in scans],
    }
    with output.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return scans


def load_manifest(
    path: Path, root: Path, verify: bool = True, expected_dataset: str | None = None
) -> list[Scan]:
    payload = json.loads(path.read_text())
    dataset = payload.get("dataset")
    if (
        payload.get("schema_version") != 1
        or dataset not in DATASET_MODALITIES
        or payload.get("modalities") != DATASET_MODALITIES[dataset]
        or payload.get("cohort_status") != "health_unverified"
    ):
        raise ValueError("Unsupported dataset manifest")
    if expected_dataset is not None and dataset != expected_dataset:
        raise ValueError(
            f"Dataset/manifest mismatch: selected {expected_dataset}, but {path} describes {dataset}. "
            "Select the matching dataset, data_root and manifest together."
        )
    scans = [Scan(**{**row, "shape": tuple(row["shape"])}) for row in payload["scans"]]
    subjects: dict[str, str] = {}
    sessions: set[tuple[str, str]] = set()
    paths: set[str] = set()
    for scan in scans:
        if scan.split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {scan.split}")
        if subjects.setdefault(scan.subject_id, scan.split) != scan.split:
            raise ValueError(f"Subject leakage across splits: {scan.subject_id}")
        key = (scan.subject_id, scan.session_id)
        if key in sessions:
            raise ValueError(f"Duplicate subject/session: {key}")
        sessions.add(key)
        if len(scan.shape) != 3 or any(size < 1 for size in scan.shape):
            raise ValueError(f"Invalid scan shape: {scan.shape}")
        files = [(scan.t1, scan.t1_sha256)]
        if dataset == "BOBs":
            if scan.t2 is None or scan.t2_sha256 is None:
                raise ValueError("BOBs manifest requires both T1 and T2")
            files.append((scan.t2, scan.t2_sha256))
        elif scan.t2 is not None or scan.t2_sha256 is not None:
            raise ValueError("IXI manifest must be T1-only")
        for relative, expected in files:
            if relative in paths or Path(relative).parts[:2] != key:
                raise ValueError(f"Duplicate or incorrectly grouped scan: {relative}")
            paths.add(relative)
            try:
                image = resolve_scan(root, relative)
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"{dataset} manifest {path} references {relative}, which is missing under {root}. "
                    "Changing data_root relocates a dataset; it does not select a new dataset. "
                    "Select the matching dataset, data_root and manifest together."
                ) from error
            if verify and sha256(image) != expected:
                raise ValueError(f"Dataset changed since freezing: {relative}")
    if {scan.split for scan in scans} != {"train", "val", "test"}:
        raise ValueError("Manifest must have nonempty train, val and test splits")
    return scans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_MODALITIES), default="BOBs")
    parser.add_argument("--data-root", type=Path, default=Path("/data/bobs"))
    parser.add_argument("--output", type=Path, default=Path("eval/frozen_manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check", action="store_true", help="Verify existing manifest and file hashes"
    )
    args = parser.parse_args()
    scans = (
        load_manifest(args.output, args.data_root, expected_dataset=args.dataset)
        if args.check
        else freeze(args.data_root, args.output, args.seed, args.dataset)
    )
    for split in ("train", "val", "test"):
        selected = [scan for scan in scans if scan.split == split]
        print(f"{split}: {len({s.subject_id for s in selected})} subjects, {len(selected)} visits")
    print(f"Manifest SHA256: {sha256(args.output)}")


if __name__ == "__main__":
    main()
