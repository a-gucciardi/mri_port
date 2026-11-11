import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from cvproj.config import RunConfig
from cvproj.data.dataset import MRISlices
from cvproj.data.prepare import freeze, load_manifest, load_pair, resolve_scan
from cvproj.data.transforms import normalize_pair, resize_slice, slice_indices
from cvproj.runtime import datasets, evaluation_protocol, reconstruction_error


def test_subject_splits_and_integrity(bobs_root: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    scans = freeze(bobs_root, manifest)
    assert len(scans) == 11
    assert {scan.split for scan in scans} == {"train", "val", "test"}
    for subject in {scan.subject_id for scan in scans}:
        assert len({scan.split for scan in scans if scan.subject_id == subject}) == 1
    assert load_manifest(manifest, bobs_root) == scans
    with pytest.raises(FileExistsError):
        freeze(bobs_root, manifest)
    other = tmp_path / "same_seed.json"
    freeze(bobs_root, other)
    assert manifest.read_bytes() == other.read_bytes()
    with (bobs_root / scans[0].t1).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Dataset changed"):
        load_manifest(manifest, bobs_root)


def test_rejects_leakage(bobs_root: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    freeze(bobs_root, manifest)
    payload = json.loads(manifest.read_text())
    payload["scans"][1]["split"] = "val" if payload["scans"][0]["split"] != "val" else "train"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Subject leakage"):
        load_manifest(manifest, bobs_root)


def test_unpaired_scans_fail(bobs_root: Path, tmp_path: Path) -> None:
    next(bobs_root.rglob("*_T2w.nii.gz")).unlink()
    with pytest.raises(ValueError, match="Unpaired"):
        freeze(bobs_root, tmp_path / "manifest.json")


def test_geometry_mismatch_fails(bobs_root: Path) -> None:
    t1 = next(bobs_root.rglob("*_T1w.nii.gz"))
    t2 = t1.with_name(t1.name.replace("_T1w", "_T2w"))
    image = nib.load(t2)
    affine = image.affine.copy()
    affine[0, 3] += 1
    nib.save(nib.Nifti1Image(image.get_fdata(), affine), t2)
    with pytest.raises(ValueError, match="physical grid"):
        load_pair(t1, t2)


def test_transforms_and_dataset(bobs_root: Path, tmp_path: Path) -> None:
    scans = freeze(bobs_root, tmp_path / "manifest.json")
    pair = normalize_pair(load_pair(bobs_root / scans[0].t1, bobs_root / scans[0].t2))
    assert pair.dtype == np.float32
    assert pair.min() >= 0 and pair.max() <= 1
    assert slice_indices(28, 4) == [5, 9, 13, 17, 21]
    resized = resize_slice(pair, 10, 32)
    assert resized.shape == (2, 32, 32)
    data = MRISlices(bobs_root, scans, 32, 4)
    for index in range(len(data)):
        image, subject = data[index]
        assert image.shape == (2, 32, 32) and subject.startswith("sub-")
        assert len(data.cache) <= 2


def test_constant_channel_and_empty_slice_fail() -> None:
    with pytest.raises(ValueError, match="intensity range"):
        normalize_pair(np.ones((2, 5, 5, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="foreground"):
        reconstruction_error(torch.zeros(1, 2, 16, 16), torch.zeros(1, 2, 16, 16))


def test_paths_cannot_escape_data_root(bobs_root: Path) -> None:
    with pytest.raises(ValueError, match="within data root"):
        resolve_scan(bobs_root, "../outside.nii.gz")


def test_ixi_single_channel_manifest_and_loading(ixi_root: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "ixi_manifest.json"
    scans = freeze(ixi_root, manifest, dataset="IXI")
    payload = json.loads(manifest.read_text())
    assert payload["dataset"] == "IXI"
    assert payload["modalities"] == ["T1w"]
    assert len(scans) == 11
    assert all(scan.t2 is None and scan.t2_sha256 is None for scan in scans)
    assert len({scan.split for scan in scans if scan.subject_id == "sub-000"}) == 1
    assert load_manifest(manifest, ixi_root, expected_dataset="IXI") == scans
    repeat = tmp_path / "repeat.json"
    freeze(ixi_root, repeat, dataset="IXI")
    assert manifest.read_bytes() == repeat.read_bytes()
    config = RunConfig(
        dataset_name="IXI", data_root=str(ixi_root), manifest=str(manifest), image_size=32
    )
    selected, manifest_hash = datasets(config, ("train", "val"))
    assert selected["train"][0][0].shape == (1, 32, 32)
    assert selected["val"][0][0].shape == (1, 32, 32)
    assert evaluation_protocol(config, manifest_hash, "val")["modalities"] == ["T1w"]
    with (ixi_root / scans[0].t1).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Dataset changed"):
        load_manifest(manifest, ixi_root, expected_dataset="IXI")


def test_wrong_dataset_manifest_is_rejected_before_file_access(
    bobs_root: Path, tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.json"
    freeze(bobs_root, manifest)
    with pytest.raises(ValueError, match="Dataset/manifest mismatch"):
        load_manifest(manifest, tmp_path / "wrong_root", expected_dataset="IXI")


def test_wrong_root_has_actionable_error(bobs_root: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    freeze(bobs_root, manifest)
    with pytest.raises(FileNotFoundError, match="does not select a new dataset"):
        load_manifest(manifest, tmp_path / "wrong_root")


def test_ixi_requires_t1_only(ixi_root: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    freeze(ixi_root, manifest, dataset="IXI")
    payload = json.loads(manifest.read_text())
    payload["scans"][0]["t2"] = payload["scans"][0]["t1"]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="must be T1-only"):
        load_manifest(manifest, ixi_root)


def test_ixi_filename_must_match_subject(ixi_root: Path, tmp_path: Path) -> None:
    image = next(ixi_root.rglob("*.nii.gz"))
    image.rename(image.with_name("msub-999_ses-00_T1w_brain_affine_mni.nii.gz"))
    with pytest.raises(ValueError, match="does not match subject/session"):
        freeze(ixi_root, tmp_path / "manifest.json", dataset="IXI")
