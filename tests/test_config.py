import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sdn_exam_demo.config import ConfigError, load_config


def test_load_config_parses_schedule():
    config = load_config(Path("config/demo_config.json"))
    assert config.exam_portal_ips == ["10.0.0.200"]
    assert config.policy.minimum_service_floor_percent == 10
    assert config.exam_schedule[0].prestage_start == datetime(2026, 4, 25, 9, 40, 0, tzinfo=UTC)


def test_current_window_matches_prestage():
    config = load_config(Path("config/demo_config.json"))
    now = datetime(2026, 4, 25, 9, 45, 0)
    assert config.current_window(now) is not None


def test_load_config_rejects_missing_required_fields(tmp_path: Path):
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps({"exam_portal_ips": ["10.0.0.200"]}), encoding="utf-8")

    with pytest.raises(ConfigError, match="missing required"):
        load_config(config_path)


def test_load_config_rejects_invalid_percent(tmp_path: Path):
    raw = json.loads(Path("config/demo_config.json").read_text(encoding="utf-8"))
    raw["policy"]["minimum_service_floor_percent"] = 101
    config_path = tmp_path / "bad-percent.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="between 1 and 100"):
        load_config(config_path)
