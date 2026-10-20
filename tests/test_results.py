from pathlib import Path

import pytest

from sdn_exam_demo.results import (
    FAILURE_FIELDNAMES,
    FailureRecord,
    MetricRecord,
    failure_reason_from_logs,
    newest_failed_log,
    write_failure_record,
    write_metric_records,
)


def test_write_metric_records_writes_schema(tmp_path: Path):
    output = write_metric_records(
        [MetricRecord(scenario="normal", metric="rtt", value=1.2, unit="ms", subject="exam")],
        tmp_path / "metrics.csv",
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "scenario,metric,value,unit,subject"
    assert "normal,rtt,1.2,ms,exam" in lines[1]


def test_write_metric_records_rejects_empty_records(tmp_path: Path):
    with pytest.raises(ValueError, match="At least one"):
        write_metric_records([], tmp_path / "metrics.csv")


def test_write_failure_record_writes_diagnostic_schema(tmp_path: Path):
    output = write_failure_record(
        FailureRecord(scenario="protected", protected=True, phase="exam_iperf", error="iperf failed"),
        tmp_path / "failure_summary.csv",
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(FAILURE_FIELDNAMES)
    assert "protected,True,exam_iperf,iperf failed" in lines[1]


def test_failure_reason_from_logs_reads_failed_run(tmp_path: Path):
    log_path = tmp_path / "failed.log"
    log_path.write_text(
        "\n".join(
            [
                "scenario=protected",
                "status=failed",
                "phase=background_iperf",
                "error=iperf3 failed",
            ]
        ),
        encoding="utf-8",
    )

    reason = failure_reason_from_logs(tmp_path)

    assert reason is not None
    assert "phase=background_iperf" in reason
    assert "error=iperf3 failed" in reason


def test_newest_failed_log_finds_latest_failed_attempt(tmp_path: Path):
    first = tmp_path / "failed" / "20260101T000000Z_protected" / "first.log"
    second = tmp_path / "failed" / "20260102T000000Z_protected" / "second.log"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("status=failed\n", encoding="utf-8")
    second.write_text("status=failed\n", encoding="utf-8")

    assert newest_failed_log(tmp_path / "latest") == second
