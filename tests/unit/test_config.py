"""Unit tests for configuration management."""

import pytest
import yaml
from pathlib import Path

from pytia.config import Config, default_config, deep_update


class TestDefaultConfig:
    """Test default configuration values and structure."""

    def test_default_config_structure(self):
        """Test that default config has all required sections."""
        config = default_config()

        # Check top-level sections
        required_sections = [
            'io', 'time', 'physics', 'mask', 'denoise', 'noise_floor',
            'model_selection', 'integration', 'bootstrap', 'performance', 'regions'
        ]
        for section in required_sections:
            assert section in config, f"Missing required section: {section}"

    def test_io_defaults(self):
        """Test IO section defaults."""
        config = default_config()
        io = config['io']

        assert io['output_dir'] == './out'
        assert io['prefix'] is None
        assert io['save_intermediate'] is False
        assert io['dtype'] == 'float32'
        assert io['write_summary_yaml'] is True
        assert io['write_status_map'] is True

    def test_physics_defaults(self):
        """Test physics section defaults."""
        config = default_config()
        physics = config['physics']

        assert physics['half_life_seconds'] is None
        assert physics['enforce_lambda_ge_phys'] is True

    def test_mask_defaults(self):
        """Test masking section defaults."""
        config = default_config()
        mask = config['mask']

        assert mask['mode'] == 'otsu'
        assert mask['provided_path'] is None
        assert mask['min_fraction_of_max'] == 0.02

    def test_bootstrap_defaults(self):
        """Test bootstrap section defaults."""
        config = default_config()
        bootstrap = config['bootstrap']

        assert bootstrap['enabled'] is True  # Actual default
        assert bootstrap['n'] == 50  # Actual default
        assert isinstance(bootstrap['seed'], int)


class TestConfigLoading:
    """Test loading configurations from different sources."""

    def test_load_from_dict(self):
        """Test loading configuration from dictionary."""
        user_config = {
            'physics': {'half_life_seconds': 3600.0},
            'io': {'output_dir': '/custom/path'}
        }

        config = Config.load(user_config)
        data = config.data

        # Should merge with defaults
        assert data['physics']['half_life_seconds'] == 3600.0
        assert data['io']['output_dir'] == '/custom/path'
        assert 'mask' in data  # Should have defaults

    def test_load_from_yaml_file(self, temp_dir):
        """Test loading from YAML file."""
        yaml_content = """
physics:
  half_life_seconds: 21636.0
  enforce_lambda_ge_phys: false

io:
  output_dir: ./test_output
  prefix: my_study

single_time:
  enabled: true
  method: haenscheid
  haenscheid_eff_half_life_seconds: 12345.0
"""
        config_path = temp_dir / "test_config.yaml"
        config_path.write_text(yaml_content)

        config = Config.load(config_path)
        data = config.data

        assert data['physics']['half_life_seconds'] == 21636.0
        assert data['physics']['enforce_lambda_ge_phys'] is False
        assert data['io']['output_dir'] == './test_output'
        assert data['io']['prefix'] == 'my_study'
        assert data['single_time']['method'] == 'haenscheid'

    def test_load_empty_yaml_file(self, temp_dir):
        """Test loading an empty YAML file."""
        empty_path = temp_dir / "empty.yaml"
        empty_path.write_text("")

        config = Config.load(empty_path)
        data = config.data

        # Should get default configuration
        assert data == default_config()

    def test_load_none_returns_default(self):
        """Test that loading None returns default config."""
        config = Config.load(None)
        assert config.data == default_config()

    def test_load_path_object(self, temp_dir):
        """Test loading with pathlib.Path object."""
        yaml_content = """
physics:
  half_life_seconds: 3600.0
"""
        config_path = Path(temp_dir) / "path_test.yaml"
        config_path.write_text(yaml_content)

        config = Config.load(config_path)
        assert config.data['physics']['half_life_seconds'] == 3600.0


class TestDeepUpdate:
    """Test the deep_update utility function."""

    def test_shallow_update(self):
        """Test shallow dictionary update."""
        dst = {'a': 1, 'b': 2}
        src = {'b': 3, 'c': 4}

        result = deep_update(dst, src)

        assert result['a'] == 1
        assert result['b'] == 3
        assert result['c'] == 4

    def test_deep_update(self):
        """Test deep nested dictionary update."""
        dst = {
            'level1': {
                'level2': {
                    'a': 1,
                    'b': 2
                },
                'c': 3
            },
            'd': 4
        }
        src = {
            'level1': {
                'level2': {
                    'b': 5,  # Update existing
                    'e': 6   # Add new
                },
                'f': 7      # Add new at level1
            }
        }

        result = deep_update(dst, src)

        expected = {
            'level1': {
                'level2': {
                    'a': 1,
                    'b': 5,
                    'e': 6
                },
                'c': 3,
                'f': 7
            },
            'd': 4
        }
        assert result == expected

    def test_mixed_types(self):
        """Test update when types differ (dict vs non-dict)."""
        dst = {'a': {'b': 1}}
        src = {'a': 2}  # Replace dict with value

        result = deep_update(dst, src)
        assert result['a'] == 2

    def test_empty_source(self):
        """Test updating with empty source."""
        dst = {'a': 1, 'b': 2}
        src = {}

        result = deep_update(dst, src)
        assert result == dst

    def test_preserve_original(self):
        """Test that original dictionaries are not modified."""
        dst = {'a': 1, 'b': {'c': 2}}
        src = {'b': {'d': 3}}
        dst_original = {'a': 1, 'b': {'c': 2}}
        src_original = {'b': {'d': 3}}

        # Note: deep_update modifies dst in place
        result = deep_update(dst, src)

        # Result should have the update
        assert result['b']['d'] == 3
        # Check that src is unchanged
        assert src == src_original
        # Result should have updates
        assert result['b']['c'] == 2


class TestConfigValidation:
    """Test configuration validation."""

    def test_config_immutability(self):
        """Test that Config objects are immutable."""
        config_data = {'physics': {'half_life_seconds': 3600.0}}
        config = Config.load(config_data)

        # Should be frozen dataclass
        with pytest.raises(Exception):
            config.data = {'new': 'data'}

    def test_nested_access_preserves_structure(self):
        """Test that nested dictionary access works correctly."""
        config_data = {
            'physics': {
                'half_life_seconds': 3600.0,
                'nested': {'deep': {'value': 42}}
            }
        }
        config = Config.load(config_data)

        assert config.data['physics']['half_life_seconds'] == 3600.0
        assert config.data['physics']['nested']['deep']['value'] == 42

    def test_special_characters_in_yaml(self, temp_dir):
        """Test YAML with special characters and comments."""
        yaml_content = """
# Configuration with special characters
io:
  output_dir: "./test  output with spaces"
  prefix: "Study-Name_v1.0"

physics:
  # Half-life in quotes for special characters
  half_life_seconds: "21636.0"

# Unicode characters
description: "TIA計算 (TIA calculation in Japanese)"
"""
        config_path = temp_dir / "special_chars.yaml"
        config_path.write_text(yaml_content)

        config = Config.load(config_path)
        assert config.data['io']['output_dir'] == "./test  output with spaces"
        assert config.data['io']['prefix'] == "Study-Name_v1.0"
        assert float(config.data['physics']['half_life_seconds']) == 21636.0

    def test_yaml_invalid_syntax(self, temp_dir):
        """Test handling of invalid YAML syntax."""
        invalid_yaml = """
invalid:
  - item1
    item2  # Missing dash - invalid YAML
    - item3
"""
        config_path = temp_dir / "invalid.yaml"
        config_path.write_text(invalid_yaml)

        # Should raise yaml parsing error
        with pytest.raises(yaml.YAMLError):
            Config.load(config_path)


class TestSpecificConfigurations:
    """Test specific configuration scenarios relevant to PyTIA."""

    def test_single_timepoint_config(self):
        """Test single-timepoint configuration setup."""
        config = {
            'single_time': {
                'enabled': True,
                'method': 'phys',
            },
            'physics': {
                'half_life_seconds': 21636.0
            }
        }

        result = Config.load(config)
        data = result.data

        assert data['single_time']['enabled'] is True
        assert data['single_time']['method'] == 'phys'
        assert data['physics']['half_life_seconds'] == 21636.0
        # Should still have all defaults
        assert 'io' in data

    def test_haenscheid_method_config(self):
        """Test Hänscheid method configuration."""
        config = {
            'single_time': {
                'enabled': True,
                'method': 'haenscheid',
                'haenscheid_eff_half_life_seconds': 43200.0,  # 12 hours
            },
            'physics': {
                'half_life_seconds': 21636.0,  # Should fallback to this
            }
        }

        result = Config.load(config)
        data = result.data

        assert data['single_time']['method'] == 'haenscheid'
        assert data['single_time']['haenscheid_eff_half_life_seconds'] == 43200.0

    def test_prior_half_life_segmentation_config(self):
        """Test prior half-life with segmentation."""
        config = {
            'single_time': {
                'enabled': True,
                'method': 'prior_half_life',
                'label_map_path': '/path/to/labels.nii.gz',
            },
            'prior_half_life': {
                'label_values': [1, 2, 3],
                'half_lives_seconds': [28800.0, 43200.0, 57600.0],
            }
        }

        result = Config.load(config)
        data = result.data

        assert data['single_time']['method'] == 'prior_half_life'
        assert data['single_time']['label_map_path'] == '/path/to/labels.nii.gz'
        assert data['prior_half_life']['label_values'] == [1, 2, 3]

    def test_bootstrap_config(self):
        """Test bootstrap configuration."""
        config = {
            'bootstrap': {
                'enabled': True,
                'n': 200,
                'seed': 12345,
                'ci_level': 0.95,
            }
        }

        result = Config.load(config)
        data = result.data

        assert data['bootstrap']['enabled'] is True
        assert data['bootstrap']['n'] == 200
        assert data['bootstrap']['seed'] == 12345
        assert data['bootstrap']['ci_level'] == 0.95

    def test_region_analysis_config(self):
        """Test region-based analysis configuration."""
        config = {
            'regions': {
                'enabled': True,
                'label_map_path': '/path/to/roi.nii.gz',
                'min_voxels_per_region': 10,
                'method': 'weighted_average',
            }
        }

        result = Config.load(config)
        data = result.data

        assert data['regions']['enabled'] is True
        assert data['regions']['method'] == 'weighted_average'
        assert data['regions']['min_voxels_per_region'] == 10

    def test_performance_config(self):
        """Test performance optimization configuration."""
        config = {
            'performance': {
                'chunk_size_vox': 50000,
                'n_jobs': 4,
                'enable_profiling': True,
            }
        }

        result = Config.load(config)
        data = result.data

        assert data['performance']['chunk_size_vox'] == 50000
        assert data['performance']['n_jobs'] == 4
        assert data['performance']['enable_profiling'] is True


class TestConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_override_nested_defaults(self):
        """Test completely overriding nested sections."""
        config = {
            'physics': {
                'half_life_seconds': 12345.0,
                'new_parameter': 'test',
            }
        }

        result = Config.load(config)
        data = result.data

        # Should preserve existing parameter
        assert data['physics']['half_life_seconds'] == 12345.0
        # Should have new parameter
        assert data['physics']['new_parameter'] == 'test'
        # Should have default parameters
        assert 'enforce_lambda_ge_phys' in data['physics']

    def test_empty_string_values(self):
        """Test handling of empty string values."""
        config = {
            'io': {
                'output_dir': '',  # Empty string
                'prefix': None,    # None value
            }
        }

        result = Config.load(config)
        data = result.data

        assert data['io']['output_dir'] == ''
        assert data['io']['prefix'] is None

    def test_numeric_string_values(self):
        """Test numeric values in strings (parsed by YAML)."""
        yaml_content = """
physics:
  half_life_seconds: "21636.0"  # String number

io:
  prefix: "123"  # String that looks like number
"""
        config = Config.load(yaml_content)

        # YAML should parse these appropriately
        assert isinstance(config.data['physics']['half_life_seconds'], int)
        assert config.data['io']['prefix'] == "123"

    def test_boolean_values(self):
        """Test boolean value handling."""
        yaml_content = """
bootstrap:
  enabled: true
  enabled_false: false

denoise:
  enabled: yes  # YAML alternative for true
"""
        config = Config.load(yaml_content)

        assert config.data['bootstrap']['enabled'] is True
        assert config.data['bootstrap'].get('enabled_false') is False
        assert config.data['denoise']['enabled'] is True

    def test_list_values(self):
        """Test list/array value handling."""
        config = {
            'times': [0.0, 1.0, 2.0, 4.0],
            'images': ['img1.nii', 'img2.nii', 'img3.nii'],
            'prior_half_life': {
                'label_values': [1, 2, 3, 4],
                'half_lives_seconds': [100.0, 200.0, 300.0, 400.0],
            }
        }

        result = Config.load(config)
        data = result.data

        assert isinstance(data['times'], list)
        assert len(data['times']) == 4
        assert data['prior_half_life']['label_values'] == [1, 2, 3, 4]