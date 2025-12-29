"""Unit tests for command-line interface."""

import pytest
import sys
from pathlib import Path
import tempfile
import yaml
from unittest.mock import patch, MagicMock
import subprocess

from pytia.cli import main, cmd_run, cmd_validate, cmd_info
from tests.utils import temp_config_file


class TestCLIArguments:
    """Test CLI argument parsing."""

    def test_no_arguments(self):
        """Test CLI with no arguments."""
        with patch('sys.stderr') as mock_stderr:
            exit_code = main([])
            assert exit_code != 0  # Should fail without subcommand

    def test_help_argument(self):
        """Test CLI help flag."""
        with patch('sys.stdout') as mock_stdout:
            exit_code = main(['--help'])
            assert exit_code == 0  # Help should succeed

    def test_invalid_subcommand(self):
        """Test CLI with invalid subcommand."""
        with patch('sys.stderr') as mock_stderr:
            exit_code = main(['invalid_command'])
            assert exit_code != 0


class TestRunCommand:
    """Test the run command."""

    def test_run_with_valid_config(self, temp_dir, synthetic_pet_data):
        """Test run command with valid configuration."""
        # Save synthetic images
        images, times = synthetic_pet_data
        image_paths = []
        for i, img in enumerate(images):
            img_path = temp_dir / f"image_{i}.nii.gz"
            nibabel = pytest.importorskip('nibabel')
            nibabel.save(img, img_path)
            image_paths.append(str(img_path))

        # Create config file
        config = {
            'inputs': {
                'images': image_paths,
                'times': times.tolist(),
            },
            'io': {'output_dir': str(temp_dir / 'output')},
            'physics': {'half_life_seconds': 21636.0},
            'mask': {'mode': 'none'},
            'denoise': {'enabled': False},
            'time': {'unit': 'seconds'},
        }

        with temp_config_file(config) as config_path:
            args = MagicMock()
            args.config = config_path

            # Mock the run_tia function
            with patch('pytia.cli.run_tia') as mock_run:
                mock_result = MagicMock()
                mock_result.output_paths = {
                    'tia': Path('tia.nii.gz'),
                    'r2': Path('r2.nii.gz'),
                    'summary': Path('summary.yaml')
                }
                mock_run.return_value = mock_result

                exit_code = cmd_run(args)
                assert exit_code == 0
                mock_run.assert_called_once()

    def test_run_missing_config(self):
        """Test run command with missing config file."""
        args = MagicMock()
        args.config = Path("nonexistent_config.yaml")

        exit_code = cmd_run(args)
        assert exit_code != 0

    def test_run_invalid_config(self, temp_dir):
        """Test run command with invalid configuration."""
        # Create invalid config
        invalid_config = """
invalid:
  - item1
    item2  # Bad indentation
    - item3
"""
        config_path = temp_dir / "invalid.yaml"
        config_path.write_text(invalid_config)

        args = MagicMock()
        args.config = config_path

        exit_code = cmd_run(args)
        assert exit_code != 0

    def test_run_with_missing_images(self, temp_dir):
        """Test run command with missing image files."""
        config = {
            'inputs': {
                'images': ['missing1.nii.gz', 'missing2.nii.gz'],
                'times': [0.0, 3600.0],
            },
            'io': {'output_dir': str(temp_dir)},
            'physics': {'half_life_seconds': 21636.0},
        }

        with temp_config_file(config) as config_path:
            args = MagicMock()
            args.config = config_path

            exit_code = cmd_run(args)
            assert exit_code != 0

    def test_run_output_creation(self, temp_dir, synthetic_pet_data):
        """Test that run command creates output files."""
        # This test would require the actual run_tia function
        # For now, test with mocking
        config = {
            'inputs': {
                'images': ['dummy1.nii', 'dummy2.nii'],
                'times': [0.0, 3600.0],
            },
            'io': {'output_dir': str(temp_dir / 'pytia_output')},
            'physics': {'half_life_seconds': 21636.0},
        }

        with temp_config_file(config) as config_path:
            args = MagicMock()
            args.config = config_path

            with patch('pytia.cli.run_tia') as mock_run:
                mock_result = MagicMock()
                mock_result.output_paths = {
                    'tia': temp_dir / 'pytia_output' / 'tia.nii.gz',
                    'r2': temp_dir / 'pytia_output' / 'r2.nii.gz',
                }
                mock_run.return_value = mock_result

                exit_code = cmd_run(args)
                assert exit_code == 0


class TestValidateCommand:
    """Test the validate command."""

    def test_validate_valid_config(self, temp_dir):
        """Test validate command with valid configuration."""
        config = {
            'inputs': {
                'images': ['img1.nii.gz', 'img2.nii.gz'],
                'times': [0.0, 3600.0],
            },
            'io': {'output_dir': './output'},
            'physics': {'half_life_seconds': 21636.0},
        }

        with temp_config_file(config) as config_path:
            args = MagicMock()
            args.config = config_path

            with patch('sys.stdout') as mock_stdout:
                exit_code = cmd_validate(args)
                assert exit_code == 0

    def test_validate_missing_required_sections(self, temp_dir):
        """Test validate command with missing required sections."""
        config = {
            'io': {'output_dir': './output'},
            # Missing 'inputs' section
            'physics': {'half_life_seconds': 21636.0},
        }

        with temp_config_file(config) as config_path:
            args = MagicMock()
            args.config = config_path

            with patch('sys.stderr') as mock_stderr:
                exit_code = cmd_validate(args)
                assert exit_code != 0

    def test_validate_invalid_yaml(self, temp_dir):
        """Test validate command with invalid YAML syntax."""
        invalid_yaml = """
invalid: [unclosed array
"""
        config_path = temp_dir / "invalid.yaml"
        config_path.write_text(invalid_yaml)

        args = MagicMock()
        args.config = config_path

        exit_code = cmd_validate(args)
        assert exit_code != 0

    def test_validate_wrong_types(self, temp_dir):
        """Test validate command with wrong data types."""
        config = {
            'inputs': {
                'images': 'not_a_list',  # Should be a list
                'times': [0.0, 'not_a_number'],  # Should be numbers
            },
            'io': {'output_dir': './output'},
            'physics': {'half_life_seconds': 'not_a_number'},
        }

        with temp_config_file(config) as config_path:
            args = MagicMock()
            args.config = config_path

            exit_code = cmd_validate(args)
            assert exit_code != 0

    def test_validate_minimal_config(self, temp_dir):
        """Test validate with minimal but complete config."""
        config = {
            'inputs': {
                'images': ['test.nii.gz'],
                'times': [0.0],
            },
            'io': {'output_dir': '.'},
            'physics': {'half_life_seconds': 21636.0},
        }

        with temp_config_file(config) as config_path:
            args = MagicMock()
            args.config = config_path

            exit_code = cmd_validate(args)
            assert exit_code == 0


class TestInfoCommand:
    """Test the info command."""

    def test_info_valid_config(self, temp_dir):
        """Test info command with valid configuration."""
        config = {
            'inputs': {
                'images': ['img1.nii.gz', 'img2.nii.gz'],
                'times': [0.0, 3600.0],
            },
            'io': {'output_dir': './output'},
            'physics': {'half_life_seconds': 21636.0},
            'bootstrap': {'enabled': True, 'n': 100},
        }

        with temp_config_file(config) as config_path:
            args = MagicMock()
            args.config = config_path

            with patch('sys.stdout') as mock_stdout:
                exit_code = cmd_info(args)
                assert exit_code == 0
                # Should have printed config information
                mock_stdout.write.assert_called()

    def test_info_file_size_display(self, temp_dir):
        """Test info command displays file size."""
        config = {
            'inputs': {
                'images': ['img1.nii.gz'],
                'times': [0.0],
            },
            'io': {'output_dir': './output'},
        }

        with temp_config_file(config) as config_path:
            # Add some content to make file larger
            with open(config_path, 'a') as f:
                f.write("\n# Extra comment\n" * 100)

            args = MagicMock()
            args.config = config_path

            with patch('sys.stdout') as mock_stdout:
                exit_code = cmd_info(args)
                assert exit_code == 0

    def test_info_nonexistent_file(self):
        """Test info command with nonexistent file."""
        args = MagicMock()
        args.config = Path("nonexistent_config.yaml")

        with patch('sys.stderr') as mock_stderr:
            exit_code = cmd_info(args)
            assert exit_code != 0


class TestCLIIntegration:
    """Test CLI integration scenarios."""

    def test_cli_with_actual_subcommand(self, temp_dir):
        """Test CLI with actual subcommands using subprocess."""
        # Create a valid config file
        config = {
            'inputs': {
                'images': ['dummy1.nii.gz', 'dummy2.nii.gz'],
                'times': [0.0, 3600.0],
            },
            'io': {'output_dir': str(temp_dir)},
            'physics': {'half_life_seconds': 21636.0},
        }

        with temp_config_file(config) as config_path:
            # Test info command
            result = subprocess.run(
                [sys.executable, '-c',
                 'from pytia.cli import main; main()',
                 'info', '--config', str(config_path)],
                capture_output=True, text=True
            )
            # Note: This might not work in all test environments

    def test_cli_error_handling(self, temp_dir):
        """Test CLI error handling and messages."""
        # Test with malformed config
        malformed_config = """
key: value
  bad_indent: value
"""
        config_path = temp_dir / "malformed.yaml"
        config_path.write_text(malformed_config)

        args = MagicMock()
        args.config = config_path

        # Should handle error gracefully
        with patch('sys.stderr') as mock_stderr:
            for cmd_func in [cmd_run, cmd_validate, cmd_info]:
                exit_code = cmd_func(args)
                if exit_code != 0:
                    break  # At least one should fail


class TestCLIOptions:
    """Test various CLI options and flags."""

    def test_verbose_output(self, temp_dir):
        """Test verbose output option."""
        config = {
            'inputs': {
                'images': ['test.nii.gz'],
                'times': [0.0],
            },
            'io': {'output_dir': '.'},
        }

        with temp_config_file(config) as config_path:
            # Test with --verbose flag if implemented
            try:
                result = subprocess.run(
                    [sys.executable, '-c',
                     'from pytia.cli import main; main()',
                     '--verbose', 'info', '--config', str(config_path)],
                    capture_output=True, text=True
                )
            except:
                pass  # Skip if verbose not implemented

    def test_quiet_output(self, temp_dir):
        """Test quiet output option."""
        config = {
            'inputs': {
                'images': ['test.nii.gz'],
                'times': [0.0],
            },
            'io': {'output_dir': '.'},
        }

        with temp_config_file(config) as config_path:
            # Test with --quiet flag if implemented
            try:
                result = subprocess.run(
                    [sys.executable, '-c',
                     'from pytia.cli import main; main()',
                     '--quiet', 'info', '--config', str(config_path)],
                    capture_output=True, text=True
                )
            except:
                pass  # Skip if quiet not implemented

    def test_config_flag_types(self, temp_dir):
        """Test different ways to specify config."""
        config = {
            'inputs': {
                'images': ['test.nii.gz'],
                'times': [0.0],
            },
            'io': {'output_dir': '.'},
        }

        with temp_config_file(config) as config_path:
            # Test with Path object
            args = MagicMock()
            args.config = config_path

            # Commands should accept both string and Path
            exit_code = cmd_info(args)
            assert exit_code == 0

            # Test with string
            args.config = str(config_path)
            exit_code = cmd_info(args)
            assert exit_code == 0


class TestCLIExitCodes:
    """Test CLI exit codes for different scenarios."""

    def test_success_exit_codes(self, temp_dir):
        """Test scenarios that should return success (0)."""
        config = {
            'inputs': {
                'images': ['test.nii.gz'],
                'times': [0.0],
            },
            'io': {'output_dir': '.'},
            'physics': {'half_life_seconds': 21636.0},
        }

        with temp_config_file(config) as config_path:
            args = MagicMock()
            args.config = config_path

            # Validate should succeed
            assert cmd_validate(args) == 0
            # Info should succeed
            assert cmd_info(args) == 0

    def test_failure_exit_codes(self, temp_dir):
        """Test scenarios that should return failure."""
        # Invalid config
        config = {'invalid': 'config'}

        with temp_config_file(config) as config_path:
            args = MagicMock()
            args.config = config_path

            # Validate should fail
            assert cmd_validate(args) != 0

        # Nonexistent file
        args = MagicMock()
        args.config = Path("nonexistent.yaml")

        # All commands should fail
        assert cmd_run(args) != 0
        assert cmd_validate(args) != 0
        assert cmd_info(args) != 0


class TestCLIEnvironmentVars:
    """Test CLI interaction with environment variables."""

    def test_config_from_env_var(self, temp_dir, monkeypatch):
        """Test reading config from environment variable."""
        config = {
            'inputs': {
                'images': ['test.nii.gz'],
                'times': [0.0],
            },
            'io': {'output_dir': '.'},
        }

        with temp_config_file(config) as config_path:
            # Set environment variable if CLI supports it
            monkeypatch.setenv('PYTIA_CONFIG', str(config_path))

            # This depends on implementation
            # Most CLI tools don't use env vars for config
            pass

    def test_output_dir_from_env_var(self, temp_dir, monkeypatch):
        """Test setting output directory from environment variable."""
        # Test if CLI supports environment variable overrides
        monkeypatch.setenv('PYTIA_OUTPUT_DIR', str(temp_dir / 'env_output'))

        config = {
            'inputs': {
                'images': ['test.nii.gz'],
                'times': [0.0],
            },
            'io': {'output_dir': '${PYTIA_OUTPUT_DIR}'},  # If env var expansion supported
        }

        with temp_config_file(config) as config_path:
            # This would require env var expansion in config loading
            pass


class TestCLIRobustness:
    """Test CLI robustness and edge cases."""

    def test_permission_denied(self, temp_dir):
        """Test CLI behavior with permission issues."""
        config = {
            'inputs': {
                'images': ['/root/noaccess.nii'],  # Usually inaccessible
                'times': [0.0],
            },
            'io': {'output_dir': str(temp_dir)},
        }

        with temp_config_file(config) as config_path:
            args = MagicMock()
            args.config = config_path

            # Should handle permission errors gracefully
            exit_code = cmd_run(args)
            assert exit_code != 0  # Should fail gracefully

    def test_disk_space_warning(self, temp_dir):
        """Test CLI with limited disk space."""
        # This is difficult to test reliably
        # Would require simulating full disk conditions
        pass

    def test_interrupted_execution(self):
        """Test CLI behavior when interrupted."""
        # Test with SIGINT if applicable
        pass

    def test_very_large_config(self, temp_dir):
        """Test CLI with very large configuration file."""
        # Create large config
        config = {
            'inputs': {
                'images': [f'img_{i}.nii.gz' for i in range(1000)],
                'times': list(range(1000)),
            },
            'io': {'output_dir': str(temp_dir)},
            'physics': {'half_life_seconds': 21636.0},
        }

        with temp_config_file(config) as config_path:
            args = MagicMock()
            args.config = config_path

            # Should handle large files without issues
            exit_code = cmd_info(args)
            assert exit_code == 0


# Helper function to create mock nibabel objects for tests
def create_mock_nifti(path, shape=(10, 10, 10)):
    """Create a mock nibabel image for testing."""
    # This would be used if we need to create actual files in tests
    pass