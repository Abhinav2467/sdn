from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from ipaddress import ip_address, ip_interface
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the demo configuration is invalid."""


@dataclass(frozen=True)
class ExamWindow:
    name: str
    start: datetime
    end: datetime
    prestage_minutes: int

    @property
    def prestage_start(self) -> datetime:
        return self.start - timedelta(minutes=self.prestage_minutes)

    def is_active(self, now: datetime) -> bool:
        return self.start <= normalize_datetime(now) <= self.end

    def is_prestage(self, now: datetime) -> bool:
        now = normalize_datetime(now)
        return self.prestage_start <= now < self.start


@dataclass(frozen=True)
class PolicyConfig:
    congestion_threshold_mbps: float
    minimum_service_floor_percent: int
    max_background_new_flows: int
    min_background_new_flows: int
    target_rtt_ms: float
    target_packet_loss_percent: float
    priority_queue_id: int
    best_effort_queue_id: int


@dataclass(frozen=True)
class DemoConfig:
    exam_portal_ips: list[str]
    exam_schedule: list[ExamWindow]
    policy: PolicyConfig
    topology: dict[str, Any]
    traffic: dict[str, Any]

    def current_window(self, now: datetime) -> ExamWindow | None:
        now = normalize_datetime(now)
        for window in self.exam_schedule:
            if window.is_active(now) or window.is_prestage(now):
                return window
        return None


def load_config(path: str | Path) -> DemoConfig:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file is not valid JSON: {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a JSON object")
    _require_keys(raw, {"exam_portal_ips", "exam_schedule", "policy", "topology", "traffic"}, "config")

    exam_portal_ips = _parse_ip_list(raw["exam_portal_ips"], "exam_portal_ips")
    exam_schedule = [_parse_exam_window(item) for item in _list(raw["exam_schedule"], "exam_schedule")]
    policy = _parse_policy(raw["policy"])
    topology = _parse_topology(raw["topology"], exam_portal_ips)
    traffic = _parse_traffic(raw["traffic"])

    return DemoConfig(
        exam_portal_ips=exam_portal_ips,
        exam_schedule=exam_schedule,
        policy=policy,
        topology=topology,
        traffic=traffic,
    )


def normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_exam_window(raw: dict[str, Any]) -> ExamWindow:
    _require_keys(raw, {"name", "start", "end", "prestage_minutes"}, "exam_schedule item")
    start = _parse_datetime(raw["start"], "exam_schedule.start")
    end = _parse_datetime(raw["end"], "exam_schedule.end")
    prestage_minutes = _positive_int(raw["prestage_minutes"], "exam_schedule.prestage_minutes", minimum=0)
    if end <= start:
        raise ConfigError("exam_schedule.end must be after exam_schedule.start")
    return ExamWindow(
        name=str(raw["name"]),
        start=start,
        end=end,
        prestage_minutes=prestage_minutes,
    )


def _parse_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigError(f"{field_name} must be a valid ISO-8601 datetime") from exc
    return normalize_datetime(parsed)


def _parse_policy(raw: Any) -> PolicyConfig:
    raw = _dict(raw, "policy")
    _require_keys(
        raw,
        {
            "congestion_threshold_mbps",
            "minimum_service_floor_percent",
            "max_background_new_flows",
            "min_background_new_flows",
            "target_rtt_ms",
            "target_packet_loss_percent",
            "priority_queue_id",
            "best_effort_queue_id",
        },
        "policy",
    )
    floor = _positive_int(raw["minimum_service_floor_percent"], "policy.minimum_service_floor_percent")
    if floor > 100:
        raise ConfigError("policy.minimum_service_floor_percent must be between 1 and 100")
    return PolicyConfig(
        congestion_threshold_mbps=_positive_float(
            raw["congestion_threshold_mbps"], "policy.congestion_threshold_mbps"
        ),
        minimum_service_floor_percent=floor,
        max_background_new_flows=_positive_int(
            raw["max_background_new_flows"], "policy.max_background_new_flows"
        ),
        min_background_new_flows=_positive_int(
            raw["min_background_new_flows"], "policy.min_background_new_flows"
        ),
        target_rtt_ms=_positive_float(
            raw["target_rtt_ms"], "policy.target_rtt_ms"
        ),
        target_packet_loss_percent=_positive_float(
            raw["target_packet_loss_percent"], "policy.target_packet_loss_percent"
        ),
        priority_queue_id=_positive_int(raw["priority_queue_id"], "policy.priority_queue_id", minimum=0),
        best_effort_queue_id=_positive_int(raw["best_effort_queue_id"], "policy.best_effort_queue_id", minimum=0),
    )


def _parse_topology(raw: Any, exam_portal_ips: list[str]) -> dict[str, Any]:
    topology = dict(_dict(raw, "topology"))
    _require_keys(
        topology,
        {"exam_clients", "background_clients", "switch_bandwidth_mbps", "exam_server_ip", "gateway_ip"},
        "topology",
    )
    topology["exam_clients"] = _positive_int(topology["exam_clients"], "topology.exam_clients")
    topology["background_clients"] = _positive_int(topology["background_clients"], "topology.background_clients")
    topology["switch_bandwidth_mbps"] = _positive_int(
        topology["switch_bandwidth_mbps"], "topology.switch_bandwidth_mbps"
    )
    try:
        server = str(ip_interface(str(topology["exam_server_ip"])))
        gateway = str(ip_interface(str(topology["gateway_ip"])))
    except ValueError as exc:
        raise ConfigError("topology.exam_server_ip and topology.gateway_ip must be valid CIDR interfaces") from exc
    topology["exam_server_ip"] = server
    topology["gateway_ip"] = gateway
    if server.split("/")[0] not in exam_portal_ips:
        raise ConfigError("topology.exam_server_ip must be listed in exam_portal_ips")
    return topology


def _parse_traffic(raw: Any) -> dict[str, Any]:
    traffic = dict(_dict(raw, "traffic"))
    _require_keys(
        traffic,
        {
            "exam_flow_bandwidth_mbps",
            "background_flow_bandwidth_mbps",
            "normal_duration_seconds",
            "congested_duration_seconds",
            "ping_target",
            "iperf_port_start",
        },
        "traffic",
    )
    for key in (
        "exam_flow_bandwidth_mbps",
        "background_flow_bandwidth_mbps",
        "normal_duration_seconds",
        "congested_duration_seconds",
    ):
        traffic[key] = _positive_int(traffic[key], f"traffic.{key}")
    try:
        traffic["ping_target"] = str(ip_address(str(traffic["ping_target"])))
    except ValueError as exc:
        raise ConfigError("traffic.ping_target must be a valid IP address") from exc
    traffic["iperf_port_start"] = _port(traffic["iperf_port_start"], "traffic.iperf_port_start")
    return traffic


def _parse_ip_list(raw: Any, field_name: str) -> list[str]:
    values = _list(raw, field_name)
    if not values:
        raise ConfigError(f"{field_name} must not be empty")
    try:
        return [str(ip_address(str(value))) for value in values]
    except ValueError as exc:
        raise ConfigError(f"{field_name} must contain only valid IP addresses") from exc


def _require_keys(raw: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(raw))
    if missing:
        raise ConfigError(f"{label} is missing required field(s): {', '.join(missing)}")


def _dict(raw: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{field_name} must be an object")
    return raw


def _list(raw: Any, field_name: str) -> list[Any]:
    if not isinstance(raw, list):
        raise ConfigError(f"{field_name} must be a list")
    return raw


def _positive_int(raw: Any, field_name: str, minimum: int = 1) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be an integer") from exc
    if value < minimum:
        raise ConfigError(f"{field_name} must be >= {minimum}")
    return value


def _positive_float(raw: Any, field_name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a number") from exc
    if value <= 0:
        raise ConfigError(f"{field_name} must be > 0")
    return value


def _port(raw: Any, field_name: str) -> int:
    value = _positive_int(raw, field_name)
    if value > 65535:
        raise ConfigError(f"{field_name} must be <= 65535")
    return value
