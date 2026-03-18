"""Product-level contract tests for config and CLI behavior."""

from __future__ import annotations

import numpy as np
import pytest

from pytia.cli import main
from pytia.config import Config, default_config
from pytia.engine import MODEL_SINGLE_TIME_PHYS, STATUS_OK, _build_summary
from pytia.version import __version__


def test_config_rejects_unknown_top_level_key() -> None:
    with pytest.raises(ValueError, match="Unknown config key\\(s\\) under config"):
        Config.load(
            {
                "physics": {"half_life_seconds": 3600.0},
                "io": {"output_dir": "./out"},
                "typo_section": {"enabled": True},
            }
        )


def test_config_rejects_unknown_nested_key() -> None:
    with pytest.raises(ValueError, match="Unknown config key\\(s\\) under config.physics"):
        Config.load({"physics": {"half_life_seconds": 3600.0, "halflife_seconds": 1800.0}})


def test_config_accepts_inputs_section() -> None:
    cfg = Config.load(
        {
            "inputs": {
                "images": ["a.nii.gz", "b.nii.gz"],
                "times": [0.0, 3600.0],
            },
            "io": {"output_dir": "./out"},
            "physics": {"half_life_seconds": 3600.0},
        }
    )
    assert cfg.data["inputs"]["images"] == ["a.nii.gz", "b.nii.gz"]


def test_regions_classes_reject_unknown_class_keys() -> None:
    with pytest.raises(ValueError, match="Unknown config key\\(s\\) under config.regions.classes"):
        Config.load(
            {
                "regions": {
                    "classes": {
                        "1": {
                            "class": "rising",
                            "unexpected": "value",
                        }
                    }
                }
            }
        )


def test_label_half_lives_require_integer_like_keys_and_positive_values() -> None:
    with pytest.raises(ValueError, match="integer-like labels"):
        Config.load(
            {
                "single_time": {
                    "label_half_lives": {
                        "kidney": 3600.0,
                    }
                }
            }
        )

    with pytest.raises(ValueError, match="must be > 0"):
        Config.load(
            {
                "single_time": {
                    "label_half_lives": {
                        "1": -1.0,
                    }
                }
            }
        )


def test_cli_version_flag_returns_success(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert __version__ in captured.out


def test_cli_without_subcommand_returns_error() -> None:
    assert main([]) != 0


def test_summary_reports_runtime_version() -> None:
    summary = _build_summary(
        cfg=default_config(),
        t_s=np.array([0.0], dtype=np.float64),
        vml=1.0,
        model_vol=np.array([MODEL_SINGLE_TIME_PHYS], dtype=np.uint8),
        status_vol=np.array([STATUS_OK], dtype=np.uint8),
        idx=np.array([0], dtype=np.int64),
        timing_ms={},
        enable_prof=False,
    )
    assert summary["pytia_version"] == __version__
    assert summary["model_legend"][MODEL_SINGLE_TIME_PHYS] == "single-time phys"
