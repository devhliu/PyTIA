"""PyTIA test suite package."""

import pytest

# Register custom markers
def pytest_configure(config):
    """Register custom markers for pytest."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "system: mark test as system test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_internet: mark test as requiring internet"
    )


# Common test fixtures that should be available to all tests
@pytest.fixture
def mock_nifti_data():
    """Provide mock NIfTI data for testing."""
    import numpy as np
    import nibabel as nib

    shape = (10, 10, 10)
    data = np.random.normal(100, 10, shape).astype(np.float32)
    affine = np.eye(4)
    return nib.Nifti1Image(data, affine)


@pytest.fixture
def simple_timeseries():
    """Provide simple time series for testing."""
    import numpy as np

    times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])
    activities = np.array([10.0, 50.0, 80.0, 60.0, 30.0])
    return times, activities


# Import test utilities for easier access
from .utils import (
    assert_images_close,
    generate_test_phantom,
    create_test_images_with_different_properties,
    measure_memory_usage,
    assert_status_codes,
    compute_tia_analytical,
    TestDataGenerator,
)

__all__ = [
    'pytest_configure',
    'mock_nifti_data',
    'simple_timeseries',
    'assert_images_close',
    'generate_test_phantom',
    'create_test_images_with_different_properties',
    'measure_memory_usage',
    'assert_status_codes',
    'compute_tia_analytical',
    'TestDataGenerator',
]