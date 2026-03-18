import pytest

from pytia.config import Config, default_config, validate_user_config_keys


def test_default_config_contains_expected_sections() -> None:
    cfg = default_config()
    for key in ["io", "time", "physics", "mask", "single_time", "bootstrap", "regions"]:
        assert key in cfg


def test_config_load_merges_user_overrides() -> None:
    cfg = Config.load(
        {
            "io": {"output_dir": "./tmp-out"},
            "time": {"unit": "hours"},
            "mask": {"mode": "none"},
        }
    ).data

    assert cfg["io"]["output_dir"] == "./tmp-out"
    assert cfg["time"]["unit"] == "hours"
    assert cfg["mask"]["mode"] == "none"


def test_validate_user_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown config key"):
        validate_user_config_keys({"mask": {"method": "otsu"}})


def test_regions_classes_and_single_time_labels_require_int_like_keys() -> None:
    validate_user_config_keys(
        {
            "regions": {
                "classes": {"1": {"class": "hump"}},
            },
            "single_time": {
                "label_half_lives": {"2": 3600.0},
            },
        }
    )

    with pytest.raises(ValueError, match="integer-like labels"):
        validate_user_config_keys({"regions": {"classes": {"tumor": {"class": "hump"}}}})


def test_config_requires_mask_path_when_mode_is_provided() -> None:
    with pytest.raises(ValueError, match="mask.provided_path"):
        Config.load({"mask": {"mode": "provided", "provided_path": None}})


def test_config_rejects_invalid_time_unit() -> None:
    with pytest.raises(ValueError, match="Invalid time.unit"):
        Config.load({"time": {"unit": "minutes"}})
