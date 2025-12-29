# PyTIA Test Suite

This directory contains the comprehensive test suite for PyTIA, including unit tests, integration tests, and system tests.

## Test Structure

```
tests/
├── __init__.py              # Common fixtures and utilities
├── conftest.py              # Pytest configuration and shared fixtures
├── utils.py                 # Test utility functions and helpers
├── unit/                    # Unit tests for individual modules
│   ├── __init__.py
│   ├── test_engine.py       # Core TIA calculation engine
│   ├── test_config.py       # Configuration management
│   ├── test_io.py          # Input/output operations
│   ├── test_models.py      # Kinetic model fitting
│   ├── test_classify.py    # Time-activity curve classification
│   ├── test_masking.py     # Masking operations
│   ├── test_bootstrap.py   # Uncertainty quantification
│   └── test_cli.py         # Command-line interface
├── integration/             # Integration tests (to be added)
│   └── test_workflows.py
└── system/                  # System tests (to be added)
    ├── test_performance.py
    └── test_robustness.py
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Only Unit Tests
```bash
pytest tests/unit/
```

### Run Tests with Coverage
```bash
pytest --cov=pytia --cov-report=html
```

### Run Tests Verbose
```bash
pytest -v
```

### Run Specific Test File
```bash
pytest tests/unit/test_engine.py
```

### Run Tests by Marker
```bash
pytest -m unit          # Run only unit tests
pytest -m "not slow"    # Skip slow tests
```

## Test Categories

### Unit Tests
- Test individual functions and modules in isolation
- Use mocking where necessary to avoid dependencies
- Fast execution (< 1 second per test)

### Integration Tests
- Test interactions between multiple modules
- Use real data and actual file operations
- Focus on complete workflows

### System Tests
- Test the entire application in realistic scenarios
- Include performance and scalability tests
- May require significant resources

## Test Utilities

The `utils.py` module provides common helper functions:

- `assert_images_close()`: Compare two nibabel images
- `generate_test_phantom()`: Create synthetic PET data
- `TestDataGenerator`: Class for generating different curve types
- `temp_config_file()`: Context manager for temporary configs
- `measure_memory_usage()`: Memory monitoring decorator

## Fixtures

Common test fixtures are defined in `conftest.py`:

- `synthetic_pet_data`: Multi-timepoint PET images
- `single_timepoint_data`: Single timepoint image
- `roi_mask`: Multi-label ROI mask
- `affine_matrix`: Standard affine matrix

## Coverage Goals

- Unit tests: Target 90% line coverage
- Critical paths: 100% coverage for TIA calculations
- All public APIs: Tested with multiple parameter combinations

## Adding New Tests

When adding new functionality:

1. Create unit tests in `tests/unit/test_<module>.py`
2. Add integration tests if the feature affects multiple modules
3. Update fixtures and utilities as needed
4. Add appropriate markers (e.g., `@pytest.mark.slow`)

## Mocking Guidelines

- Mock external dependencies (network, file system)
- Use `unittest.mock` for function-level mocking
- Create fixtures for complex test data
- Keep mocked behavior realistic

## Test Data

Synthetic test data is generated programmatically to:
- Ensure reproducibility
- Cover edge cases
- Avoid copyright issues with real patient data

## CI/CD Integration

The test suite is designed to run in CI/CD environments:
- No manual intervention required
- Tests complete in under 5 minutes
- All dependencies are vendored or installable via pip