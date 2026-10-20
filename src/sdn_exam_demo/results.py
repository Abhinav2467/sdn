from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

FIELDNAMES = ["scenario", "metric", "value", "unit", "subject"]
FAILURE_FIELDNAMES = ["scenario", "protected", "phase", "error", "hint"]
FAILURE_HINT = "run sudo mn -c and confirm the controller is running before retrying"


@dataclass(frozen=True)
class MetricRecord:
    scenario: str
    metric: str
    value: float
    unit: str
    subject: str


@dataclass(frozen=True)
class FailureRecord:
    scenario: str
    protected: bool
    phase: str
    error: str
    hint: str = FAILURE_HINT


def write_metric_records(records: list[MetricRecord], output_path: str | Path) -> Path:
    if not records:
        raise ValueError("At least one metric record is required")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return output


def write_failure_record(record: FailureRecord, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_FIELDNAMES)
        writer.writeheader()
        writer.writerow(asdict(record))
    return output


def failure_reason_from_logs(results_dir: str | Path) -> str | None:
    logs = sorted(Path(results_dir).glob("*.log"))
    for log_path in reversed(logs):
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "status=failed" not in lines:
            continue
        phase = next((line.removeprefix("phase=") for line in lines if line.startswith("phase=")), "unknown")
        error = next((line.removeprefix("error=") for line in lines if line.startswith("error=")), "unknown error")
        return f"{log_path}: phase={phase}; error={error}"
    return None


def newest_failed_log(results_dir: str | Path) -> Path | None:
    failed_root = Path(results_dir).parent / "failed"
    logs = sorted(failed_root.glob("*/*.log"))
    return logs[-1] if logs else None
