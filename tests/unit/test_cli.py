from pathlib import Path
from types import SimpleNamespace

import yaml

from pytia import cli


def _write_yaml(path: Path, data: dict) -> Path:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return path


def test_main_version_returns_success() -> None:
    assert cli.main(["--version"]) == 0


def test_validate_command_handles_valid_and_invalid_configs(tmp_path: Path) -> None:
    valid = _write_yaml(tmp_path / "valid.yaml", {"mask": {"mode": "none"}})
    invalid = _write_yaml(tmp_path / "invalid.yaml", {"mask": {"method": "otsu"}})

    assert cli.main(["validate", "--config", str(valid)]) == 0
    assert cli.main(["validate", "--config", str(invalid)]) == 1


def test_info_command_prints_config_file(tmp_path: Path, capsys) -> None:
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"io": {"output_dir": "./out"}})

    rc = cli.main(["info", "--config", str(cfg_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "output_dir" in out


def test_run_command_requires_inputs(tmp_path: Path) -> None:
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"mask": {"mode": "none"}})

    rc = cli.main(["run", "--config", str(cfg_path)])

    assert rc == 1


def test_run_command_calls_engine_with_valid_config(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_yaml(
        tmp_path / "run.yaml",
        {
            "io": {"output_dir": str(tmp_path / "out")},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "noise_floor": {"enabled": False},
            "bootstrap": {"enabled": False},
            "inputs": {
                "images": ["tp1.nii.gz"],
                "times": [1.0],
            },
        },
    )

    called: dict[str, object] = {}

    def _fake_run_tia(*, images, times, config, mask):
        called["images"] = images
        called["times"] = times
        called["config"] = config
        called["mask"] = mask
        return SimpleNamespace(output_paths={"tia": Path("/tmp/tia.nii.gz")})

    monkeypatch.setattr("pytia.cli.run_tia", _fake_run_tia)

    rc = cli.main(["run", "--config", str(cfg_path)])

    assert rc == 0
    assert called["images"] == ["tp1.nii.gz"]
    assert called["times"] == [1.0]
    assert called["mask"] is None
