from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def bobs_root(tmp_path: Path) -> Path:
    root = tmp_path / "bobs"
    generator = np.random.default_rng(0)
    for subject in range(10):
        for session in range(2 if subject == 0 else 1):
            folder = root / f"sub-{subject:03}" / f"ses-{session}mo" / "anat"
            folder.mkdir(parents=True)
            prefix = f"sub-{subject:03}_ses-{session}mo_space-INFANTMNIacpc"
            for modality in ("T1w", "T2w"):
                image = generator.uniform(0, 1000, (20, 24, 28)).astype(np.float32)
                nib.save(nib.Nifti1Image(image, np.eye(4)), folder / f"{prefix}_{modality}.nii.gz")
    return root
