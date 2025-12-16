"""Unit tests for single-timepoint (STP) TIA calculation methods."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import nibabel as nib

from pytia.config import Config
from pytia.engine import run_tia


def create_simple_image(shape=(10, 10, 10), value=100.0):
    """Create a simple NIfTI image with constant values."""
    data = np.full(shape, value, dtype=np.float32)
    affine = np.eye(4)
    return nib.Nifti1Image(data, affine)


def create_label_image(shape=(10, 10, 10)):
    """Create a label/ROI image with three regions."""
    data = np.zeros(shape, dtype=np.int32)
    data[0:3, :, :] = 1  # Label 1
    data[3:6, :, :] = 2  # Label 2
    data[6:, :, :] = 3   # Label 3
    affine = np.eye(4)
    return nib.Nifti1Image(data, affine)


class TestSingleTimePointPhysicalDecay:
    """Test single-timepoint method: physical decay."""

    def test_phys_method_with_valid_halflife(self):
        """Test physical decay method with valid half-life."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create single timepoint image
            img = create_simple_image(shape=(5, 5, 5), value=100.0)
            img_path = Path(tmpdir) / "activity.nii.gz"
            nib.save(img, img_path)

            config = {
                "io": {"output_dir": tmpdir},
                "physics": {"half_life_seconds": 3600.0},  # 1 hour
                "single_time": {
                    "enabled": True,
                    "method": "phys",
                },
                "time": {"unit": "seconds"},
            }

            result = run_tia([img_path], times=[0.0], config=config)

            # With lambda = ln(2)/3600, TIA = 100 / lambda = 100 * 3600 / ln(2) ≈ 519615
            expected_tia = 100.0 * 3600.0 / np.log(2.0)
            assert result.tia_img is not None
            tia_data = np.asarray(result.tia_img.dataobj)
            assert np.any(np.isfinite(tia_data))
            assert np.allclose(tia_data[2, 2, 2], expected_tia, rtol=0.01)

    def test_phys_method_missing_halflife(self):
        """Test physical decay method fails without half-life."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = create_simple_image(shape=(5, 5, 5), value=100.0)
            img_path = Path(tmpdir) / "activity.nii.gz"
            nib.save(img, img_path)

            config = {
                "io": {"output_dir": tmpdir},
                "physics": {"half_life_seconds": None},
                "single_time": {
                    "enabled": True,
                    "method": "phys",
                },
            }

            result = run_tia([img_path], times=[0.0], config=config)
            tia_data = np.asarray(result.tia_img.dataobj)
            # All non-background should be NaN
            assert np.all(~np.isfinite(tia_data[(tia_data != 0)]))


class TestSingleTimePointHaenscheid:
    """Test single-timepoint method: Hänscheid method."""

    def test_haenscheid_method_explicit_halflife(self):
        """Test Hänscheid method with explicit effective half-life."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = create_simple_image(shape=(5, 5, 5), value=100.0)
            img_path = Path(tmpdir) / "activity.nii.gz"
            nib.save(img, img_path)

            eff_hl = 7200.0  # 2 hours effective half-life
            config = {
                "io": {"output_dir": tmpdir},
                "physics": {"half_life_seconds": 3600.0},  # Physical (different from eff)
                "single_time": {
                    "enabled": True,
                    "method": "haenscheid",
                    "haenscheid_eff_half_life_seconds": eff_hl,
                },
            }

            result = run_tia([img_path], times=[0.0], config=config)
            tia_data = np.asarray(result.tia_img.dataobj)
            expected_tia = 100.0 * eff_hl / np.log(2.0)
            assert np.allclose(tia_data[2, 2, 2], expected_tia, rtol=0.01)

    def test_haenscheid_fallback_to_phys_halflife(self):
        """Test Hänscheid method falls back to physics.half_life_seconds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = create_simple_image(shape=(5, 5, 5), value=100.0)
            img_path = Path(tmpdir) / "activity.nii.gz"
            nib.save(img, img_path)

            config = {
                "io": {"output_dir": tmpdir},
                "physics": {"half_life_seconds": 3600.0},
                "single_time": {
                    "enabled": True,
                    "method": "haenscheid",
                    "haenscheid_eff_half_life_seconds": None,  # Use fallback
                },
            }

            result = run_tia([img_path], times=[0.0], config=config)
            tia_data = np.asarray(result.tia_img.dataobj)
            expected_tia = 100.0 * 3600.0 / np.log(2.0)
            assert np.allclose(tia_data[2, 2, 2], expected_tia, rtol=0.01)


class TestSingleTimePointPriorHalfLife:
    """Test single-timepoint method: prior half-life (global and label-based)."""

    def test_prior_halflife_global(self):
        """Test prior half-life method with global value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = create_simple_image(shape=(5, 5, 5), value=100.0)
            img_path = Path(tmpdir) / "activity.nii.gz"
            nib.save(img, img_path)

            config = {
                "io": {"output_dir": tmpdir},
                "single_time": {
                    "enabled": True,
                    "method": "prior_half_life",
                    "half_life_seconds": 5400.0,  # 1.5 hours
                },
            }

            result = run_tia([img_path], times=[0.0], config=config)
            tia_data = np.asarray(result.tia_img.dataobj)
            expected_tia = 100.0 * 5400.0 / np.log(2.0)
            assert np.allclose(tia_data[2, 2, 2], expected_tia, rtol=0.01)

    def test_prior_halflife_label_map(self):
        """Test prior half-life method with label-map based mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create activity image with constant values
            activity_img = create_simple_image(shape=(6, 6, 6), value=100.0)
            activity_path = Path(tmpdir) / "activity.nii.gz"
            nib.save(activity_img, activity_path)

            # Create label image with three regions
            label_img = create_label_image(shape=(6, 6, 6))
            label_path = Path(tmpdir) / "labels.nii.gz"
            nib.save(label_img, label_path)

            config = {
                "io": {"output_dir": tmpdir},
                "single_time": {
                    "enabled": True,
                    "method": "prior_half_life",
                    "label_map_path": str(label_path),
                    "half_life_seconds": 3600.0,  # default
                    "label_half_lives": {
                        1: 1800.0,   # Label 1: 30 min
                        2: 3600.0,   # Label 2: 60 min
                        3: 5400.0,   # Label 3: 90 min
                    },
                },
            }

            result = run_tia([activity_path], times=[0.0], config=config)
            tia_data = np.asarray(result.tia_img.dataobj)

            # Check different labels have different TIA values (scaled by their half-life)
            # Label 1 region (indices 0-2): hl=1800
            label1_tia = 100.0 * 1800.0 / np.log(2.0)
            assert np.allclose(tia_data[1, 1, 1], label1_tia, rtol=0.05)

            # Label 2 region (indices 3-5): hl=3600
            label2_tia = 100.0 * 3600.0 / np.log(2.0)
            assert np.allclose(tia_data[4, 4, 4], label2_tia, rtol=0.05)

            # Label 3 region (indices 6+): hl=5400
            label3_tia = 100.0 * 5400.0 / np.log(2.0)
            assert np.allclose(tia_data[5, 5, 5], label3_tia, rtol=0.05)

    def test_prior_halflife_missing_halflife(self):
        """Test prior half-life method fails when no half-life provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = create_simple_image(shape=(5, 5, 5), value=100.0)
            img_path = Path(tmpdir) / "activity.nii.gz"
            nib.save(img, img_path)

            config = {
                "io": {"output_dir": tmpdir},
                "single_time": {
                    "enabled": True,
                    "method": "prior_half_life",
                    "half_life_seconds": None,
                },
            }

            result = run_tia([img_path], times=[0.0], config=config)
            tia_data = np.asarray(result.tia_img.dataobj)
            # All should be NaN
            assert np.all(~np.isfinite(tia_data[(tia_data != 0)]))


class TestSingleTimePointNegativeValues:
    """Test single-timepoint with negative/zero values."""

    def test_negative_clamping(self):
        """Test that negative activity values are clamped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.array([[[100.0, -10.0], [50.0, -5.0]]])
            img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
            img_path = Path(tmpdir) / "activity.nii.gz"
            nib.save(img, img_path)

            config = {
                "io": {"output_dir": tmpdir},
                "physics": {"half_life_seconds": 3600.0},
                "single_time": {
                    "enabled": True,
                    "method": "phys",
                },
            }

            result = run_tia([img_path], times=[0.0], config=config)
            tia_data = np.asarray(result.tia_img.dataobj)
            # Negative values should not produce finite TIA
            assert not np.isfinite(tia_data[0, 0, 1])  # Was -10
            assert not np.isfinite(tia_data[0, 1, 1])  # Was -5


class TestSingleTimePointDisabled:
    """Test that STP is skipped when disabled."""

    def test_single_time_disabled_uses_multitime_path(self):
        """Test that when single_time.enabled=False, multi-timepoint path is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a single timepoint image
            img = create_simple_image(shape=(5, 5, 5), value=100.0)
            img_path = Path(tmpdir) / "activity.nii.gz"
            nib.save(img, img_path)

            config = {
                "io": {"output_dir": tmpdir},
                "physics": {"half_life_seconds": 3600.0},
                "single_time": {
                    "enabled": False,  # Disabled
                    "method": "phys",
                },
                "time": {"unit": "seconds"},
            }

            # Should not raise error; should just use multitime logic
            result = run_tia([img_path], times=[0.0], config=config)
            assert result.tia_img is not None


class TestSingleTimePointWithNoiseFloor:
    """Test single-timepoint with noise-floor filtering."""

    def test_noise_floor_filtering(self):
        """Test that values below noise floor are marked as invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image with mix of high and low values
            data = np.full((5, 5, 5), 1.0, dtype=np.float32)
            data[1:3, 1:3, 1:3] = 1000.0  # High activity region
            img = nib.Nifti1Image(data, np.eye(4))
            img_path = Path(tmpdir) / "activity.nii.gz"
            nib.save(img, img_path)

            config = {
                "io": {"output_dir": tmpdir},
                "physics": {"half_life_seconds": 3600.0},
                "single_time": {
                    "enabled": True,
                    "method": "phys",
                },
                "noise_floor": {
                    "enabled": True,
                    "mode": "relative",
                    "relative_fraction_of_voxel_max": 0.1,  # 10% of max
                },
            }

            result = run_tia([img_path], times=[0.0], config=config)
            tia_data = np.asarray(result.tia_img.dataobj)

            # High region should have valid TIA
            assert np.isfinite(tia_data[2, 2, 2])
            # Low region should be below floor (invalid)
            assert not np.isfinite(tia_data[0, 0, 0])


class TestSingleTimePointModelID:
    """Test that correct model IDs are assigned."""

    def test_model_id_phys(self):
        """Test model ID 101 for phys method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = create_simple_image(shape=(5, 5, 5), value=100.0)
            img_path = Path(tmpdir) / "activity.nii.gz"
            nib.save(img, img_path)

            config = {
                "io": {"output_dir": tmpdir},
                "physics": {"half_life_seconds": 3600.0},
                "single_time": {
                    "enabled": True,
                    "method": "phys",
                },
            }

            result = run_tia([img_path], times=[0.0], config=config)
            model_data = np.asarray(result.model_id_img.dataobj)
            assert model_data[2, 2, 2] == 101

    def test_model_id_haenscheid(self):
        """Test model ID 102 for haenscheid method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = create_simple_image(shape=(5, 5, 5), value=100.0)
            img_path = Path(tmpdir) / "activity.nii.gz"
            nib.save(img, img_path)

            config = {
                "io": {"output_dir": tmpdir},
                "physics": {"half_life_seconds": 3600.0},
                "single_time": {
                    "enabled": True,
                    "method": "haenscheid",
                    "haenscheid_eff_half_life_seconds": 7200.0,
                },
            }

            result = run_tia([img_path], times=[0.0], config=config)
            model_data = np.asarray(result.model_id_img.dataobj)
            assert model_data[2, 2, 2] == 102

    def test_model_id_prior(self):
        """Test model ID 103 for prior_half_life method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = create_simple_image(shape=(5, 5, 5), value=100.0)
            img_path = Path(tmpdir) / "activity.nii.gz"
            nib.save(img, img_path)

            config = {
                "io": {"output_dir": tmpdir},
                "single_time": {
                    "enabled": True,
                    "method": "prior_half_life",
                    "half_life_seconds": 5400.0,
                },
            }

            result = run_tia([img_path], times=[0.0], config=config)
            model_data = np.asarray(result.model_id_img.dataobj)
            assert model_data[2, 2, 2] == 103
