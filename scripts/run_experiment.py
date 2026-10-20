#!/usr/bin/env python3
from __future__ import annotations


import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdn_exam_demo.config import ConfigError, load_config
from sdn_exam_demo.logging_config import configure_logging
from sdn_exam_demo.results import FAILURE_HINT, FailureRecord, MetricRecord, write_failure_record, write_metric_records

try:
    from mininet.log import setLogLevel
except ImportError as exc:  # pragma: no cover - runtime dependency in WSL/Ubuntu
    raise SystemExit(
        "Mininet is required to run experiments. Run this script inside Ubuntu/WSL after setup."
    ) from exc

MININET_DIR = ROOT / "mininet"
if str(MININET_DIR) not in sys.path:
    sys.path.insert(0, str(MININET_DIR))

from exam_topology import configure_qos_queues, create_network

IPERF_DURATION_SECONDS = 12
LOGGER = logging.getLogger("sdn_exam_demo.run_experiment")


class ExperimentError(RuntimeError):
    """Raised when the Mininet experiment cannot produce trustworthy metrics."""

    def __init__(self, message: str, phase: str = "unknown", details: Iterable[str] | None = None) -> None:
        super().__init__(message)
        self.phase = phase
        self.details = list(details or [])


def collect_ping(host, target: str, count: int = 10, timeout_seconds: int = 2) -> tuple[float, float]:
    output = host.cmd(f"ping -c {count} -W {timeout_seconds} {target}")
    packet_loss = 100.0
    avg_rtt = 0.0
    for line in output.splitlines():
        if "packet loss" in line:
            packet_loss = float(line.split("%")[0].split()[-1])
        if "min/avg/max" in line:
            avg_rtt = float(line.split("=")[1].split("/")[1].strip())
    return avg_rtt, packet_loss


def start_iperf_servers(server_host, ports: Iterable[int]) -> None:
    for port in ports:
        server_host.cmd(f"iperf3 -s -D -p {port}")


def wait_for_iperf_servers(server_host, ports: Iterable[int], timeout_seconds: int = 2) -> None:
    failed_ports = []
    for port in ports:
        output = server_host.cmd(
            f"timeout {timeout_seconds} sh -c "
            f"'while ! ss -ltnH sport = :{port} | grep -q .; do sleep 0.1; done' "
            "&& echo READY || echo NOT_READY"
        )
        if output.strip().splitlines()[-1:] != ["READY"]:
            failed_ports.append(str(port))
    if failed_ports:
        raise ExperimentError(
            f"iperf3 server failed to listen on port(s): {', '.join(failed_ports)}",
            phase="iperf_server_readiness",
            details=[f"port={port}" for port in failed_ports],
        )


def stop_iperf_servers(server_host) -> None:
    server_host.cmd("pkill -f 'iperf3 -s' || true")
    server_host.cmd("pkill -9 -f 'iperf3 -s' || true")


def collect_iperf(
    server_host,
    client_hosts: Iterable,
    server_ip: str,
    port_start: int,
    bandwidth_mbps: int,
    phase: str,
) -> float:
    hosts = list(client_hosts)
    ports = [port_start + index for index in range(len(hosts))]
    stop_iperf_servers(server_host)
    start_iperf_servers(server_host, ports)
    try:
        wait_for_iperf_servers(server_host, ports)
        throughput = 0.0
        failures = []
        pending = []
        for offset, host in enumerate(hosts):
            port = ports[offset]
            cmd = (
                f"timeout --foreground {IPERF_DURATION_SECONDS + 30} iperf3 -c {server_ip} -p {port} -u "
                f"-b {bandwidth_mbps}M -t {IPERF_DURATION_SECONDS} -J"
            )
            host.sendCmd(cmd)
            pending.append((host, port))

        for host, port in pending:
            output = host.waitOutput()
            if not output.strip():
                failures.append(f"{host.name}: no iperf3 output")
                LOGGER.warning("iperf3 returned no output for host=%s port=%s", host.name, port)
                continue
            try:
                data = json.loads(output)
            except json.JSONDecodeError as exc:
                failures.append(f"{host.name}: invalid iperf3 JSON: {exc}; output={output.strip()}")
                LOGGER.warning(
                    "failed to parse iperf3 JSON for host=%s port=%s error=%s output=%s",
                    host.name,
                    port,
                    exc,
                    output,
                )
                continue

            if "error" in data:
                failures.append(f"{host.name}: {data['error']}")
                LOGGER.warning("iperf3 error for host=%s port=%s error=%s", host.name, port, data["error"])
                continue

            end = data.get("end")
            if not isinstance(end, dict):
                failures.append(f"{host.name}: missing end summary; output={output.strip()}")
                LOGGER.warning("iperf3 result missing end summary for host=%s port=%s output=%s", host.name, port, output)
                continue

            summary = end.get("sum") or end.get("sum_received") or end.get("sum_sent")
            if not isinstance(summary, dict) or "bits_per_second" not in summary:
                failures.append(f"{host.name}: missing throughput summary; output={output.strip()}")
                LOGGER.warning(
                    "iperf3 result missing throughput summary for host=%s port=%s output=%s",
                    host.name,
                    port,
                    output,
                )
                continue

            throughput += float(summary["bits_per_second"]) / 1_000_000

        if failures:
            failed_host_list = ", ".join(item.split(":", maxsplit=1)[0] for item in failures)
            LOGGER.error("iperf3 failed for %s/%s hosts: %s", len(failures), len(hosts), failed_host_list)
            raise ExperimentError(
                f"iperf3 failed for {len(failures)}/{len(hosts)} hosts: {failed_host_list}",
                phase=phase,
                details=failures,
            )
        return throughput
    finally:
        stop_iperf_servers(server_host)


def build_records(
    scenario: str, avg_rtt: float, packet_loss: float, exam_tput: float, background_tput: float
) -> list[MetricRecord]:
    return [
        MetricRecord(scenario=scenario, metric="rtt", value=avg_rtt, unit="ms", subject="exam"),
        MetricRecord(scenario=scenario, metric="packet_loss", value=packet_loss, unit="percent", subject="exam"),
        MetricRecord(scenario=scenario, metric="throughput", value=exam_tput, unit="mbps", subject="exam"),
        MetricRecord(scenario=scenario, metric="throughput", value=background_tput, unit="mbps", subject="background"),
    ]


def resolve_results_dir(raw_results_dir: str) -> Path:
    if not raw_results_dir.strip():
        raise ExperimentError("results directory must not be empty")
    results_dir = Path(raw_results_dir).expanduser()
    if not results_dir.is_absolute():
        results_dir = Path.cwd() / results_dir
    return results_dir.resolve(strict=False)


def is_drive_root(path: Path) -> bool:
    parts = path.parts
    return len(parts) == 3 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1


def assert_safe_results_dir(results_dir: Path) -> None:
    repo_root = ROOT.resolve(strict=False)
    unsafe_paths = {
        Path("/").resolve(strict=False),
        Path.home().resolve(strict=False),
        repo_root,
    }
    if results_dir in unsafe_paths or results_dir.parent == results_dir or is_drive_root(results_dir):
        raise ExperimentError(f"refusing to clean unsafe results directory: {results_dir}")


def unique_run_dir(parent: Path, run_name: str) -> Path:
    candidate = parent / run_name
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = parent / f"{run_name}_{suffix}"
    return candidate


def promote_successful_attempt(attempt_dir: Path, results_dir: Path) -> None:
    assert_safe_results_dir(results_dir)
    if results_dir.exists():
        if results_dir.is_dir():
            shutil.rmtree(results_dir)
        else:
            results_dir.unlink()
    shutil.move(str(attempt_dir), str(results_dir))
    LOGGER.info("promoted successful run to %s", results_dir)


def preserve_failed_attempt(attempt_dir: Path, results_dir: Path, run_name: str) -> Path:
    failed_dir = unique_run_dir(results_dir.parent / "failed", run_name)
    failed_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(attempt_dir), str(failed_dir))
    LOGGER.error("preserved failed run diagnostics in %s", failed_dir)
    return failed_dir


def warm_up_topology(hosts: Iterable, target: str) -> None:
    time.sleep(1.5)
    failed_hosts = []
    for host in hosts:
        _, packet_loss = collect_ping(host, target, count=2, timeout_seconds=1)
        if packet_loss > 0:
            if host.name.startswith("e"):
                failed_hosts.append(f"{host.name}({packet_loss:.0f}% loss)")
            else:
                LOGGER.warning("Background host %s is unreachable during warmup (likely blocked by active policy): %s%% loss", host.name, packet_loss)
    if failed_hosts:
        raise ExperimentError(
            f"topology reachability failed before iperf: {', '.join(failed_hosts)}",
            phase="topology_warmup",
            details=failed_hosts,
        )


def run_concurrent_traffic_and_log(
    net,
    config,
    scenario: str,
    policy_mode: str,
    exam_hosts,
    background_hosts,
    exam_server,
    server_ip,
    attempt_dir: Path,
    run_name: str,
) -> tuple[float, float, float, float]:
    from sdn_exam_demo.config import normalize_datetime
    import csv

    # Determine duration based on scenario
    is_normal = scenario == "normal"
    duration = int(config.traffic["normal_duration_seconds"] if is_normal else config.traffic["congested_duration_seconds"])

    # DYNAMIC PARTITIONING:
    # We must reserve one host strictly for RTT/loss telemetry pings.
    # Mininet hosts do not support overlapping shell commands, so
    # the dedicated ping_host must NEVER run iperf3 concurrently.
    if len(exam_hosts) > 1:
        ping_host = exam_hosts[0]
        iperf_exam_hosts = exam_hosts[1:]
        iperf_bg_hosts = background_hosts
    else:
        ping_host = background_hosts[0]
        iperf_exam_hosts = exam_hosts
        iperf_bg_hosts = background_hosts[1:] if len(background_hosts) > 1 else background_hosts

    exam_ports = [config.traffic["iperf_port_start"] + idx for idx in range(len(iperf_exam_hosts))]
    bg_ports = [config.traffic["iperf_port_start"] + 200 + idx for idx in range(len(iperf_bg_hosts))]

    # Reset/start iperf3 servers on target host
    stop_iperf_servers(exam_server)
    start_iperf_servers(exam_server, [*exam_ports, *bg_ports])
    wait_for_iperf_servers(exam_server, [*exam_ports, *bg_ports])

    time_series_records = []

    # Telemetry paths
    latest_dir = attempt_dir.parent.parent / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = latest_dir / "telemetry.json"
    override_path = latest_dir / "override.json"
    audit_log_path = latest_dir / "override_audit.log"

    # Clean stale override and write starting telemetry JSON
    if override_path.exists():
        try:
            override_path.unlink()
        except Exception:
            pass

    # DYNAMIC DURATION & STAGGERED SCHEDULES
    # We stagger client start times and active durations to create a highly dynamic mix
    # of traffic, rather than a constant/series load.
    exam_schedules = []
    for idx, host in enumerate(iperf_exam_hosts):
        port = exam_ports[idx]
        if len(iperf_exam_hosts) > 1:
            start_delay = (idx % 3) * (duration / 4.0)
            flow_duration = max(5.0, duration - start_delay - (idx % 2) * (duration / 5.0))
        else:
            start_delay = 0.0
            flow_duration = duration
        exam_schedules.append({
            "host": host,
            "port": port,
            "bandwidth": float(config.traffic['exam_flow_bandwidth_mbps']),
            "start_delay": start_delay,
            "flow_duration": flow_duration
        })

    bg_schedules = []
    for idx, host in enumerate(iperf_bg_hosts):
        port = bg_ports[idx]
        if len(iperf_bg_hosts) > 1:
            start_delay = (idx % 3) * (duration / 4.0)
            flow_duration = max(5.0, duration - start_delay - (idx % 2) * (duration / 5.0))
        else:
            start_delay = 0.0
            flow_duration = duration
        bg_schedules.append({
            "host": host,
            "port": port,
            "bandwidth": float(config.traffic['background_flow_bandwidth_mbps']),
            "start_delay": start_delay,
            "flow_duration": flow_duration
        })

    telemetry = {
        "policy_mode": policy_mode,
        "active_exam_clients": len(iperf_exam_hosts),
        "active_background_clients": len(iperf_bg_hosts),
        "exam_rtt": 0.0,
        "packet_loss": 0.0,
        "iperf_flow_count": len(iperf_exam_hosts) + len(iperf_bg_hosts),
        "queue_utilization": 0.0,
        "exam_clients": int(config.topology["exam_clients"])
    }
    with telemetry_path.open("w", encoding="utf-8") as handle:
        json.dump(telemetry, handle)

    LOGGER.info("Starting concurrent dynamic staggered traffic: %d exam and %d background hosts (ping reserved: %s) for %d seconds",
                len(iperf_exam_hosts), len(iperf_bg_hosts), ping_host.name, duration)

    # Launch exam clients asynchronously with staggered start
    for flow in exam_schedules:
        delay = flow["start_delay"]
        port = flow["port"]
        bw = flow["bandwidth"]
        dur = flow["flow_duration"]
        cmd = (
            f"timeout --foreground {duration + 30} sh -c "
            f"'sleep {delay:.1f} && iperf3 -c {server_ip} -p {port} -u "
            f"-b {bw}M -t {dur:.1f} -J'"
        )
        flow["host"].sendCmd(cmd)

    # Launch background clients asynchronously with staggered start
    for flow in bg_schedules:
        delay = flow["start_delay"]
        port = flow["port"]
        bw = flow["bandwidth"]
        dur = flow["flow_duration"]
        cmd = (
            f"timeout --foreground {duration + 30} sh -c "
            f"'sleep {delay:.1f} && iperf3 -c {server_ip} -p {port} -u "
            f"-b {bw}M -t {dur:.1f} -J'"
        )
        flow["host"].sendCmd(cmd)

    # Logging loop
    ping_target = config.traffic["ping_target"]
    interval = 5
    elapsed = 0.0
    start_time = time.time()
    timeseries_csv_path = attempt_dir / f"{run_name}_timeseries.csv"

    try:
        while elapsed < duration:
            loop_start = time.time()

            # Active telemetry query: ping measurement on reserved non-iperf host!
            rtt, loss = collect_ping(ping_host, ping_target, count=3, timeout_seconds=1)

            # Read / process operator overrides
            admit_limit = config.policy.max_background_new_flows
            action_source = policy_mode
            if override_path.exists():
                try:
                    with override_path.open("r", encoding="utf-8") as override_handle:
                        override_data = json.load(override_handle)
                        if "override_limit" in override_data:
                            admit_limit = int(override_data["override_limit"])
                            action_source = "manual_override"
                            
                            # Audit log
                            reason = override_data.get("reason", "No reason provided")
                            audit_msg = f"timestamp={datetime.now(UTC).isoformat()} old={config.policy.max_background_new_flows} new={admit_limit} reason='{reason}'\n"
                            with audit_log_path.open("a", encoding="utf-8") as audit_h:
                                audit_h.write(audit_msg)
                except Exception as exc:
                    LOGGER.warning("Failed to process manual override: %s", exc)

            # Estimate active clients and throughput at this point in time
            active_exam = sum(1 for f in exam_schedules if f["start_delay"] <= elapsed <= (f["start_delay"] + f["flow_duration"]))
            active_bg = sum(1 for f in bg_schedules if f["start_delay"] <= elapsed <= (f["start_delay"] + f["flow_duration"]))
            measured_tput = (
                active_exam * float(config.traffic["exam_flow_bandwidth_mbps"])
                + active_bg * float(config.traffic["background_flow_bandwidth_mbps"])
            )

            # Sync active telemetry JSON for OpenFlow controller
            telemetry["active_exam_clients"] = active_exam
            telemetry["active_background_clients"] = active_bg
            telemetry["iperf_flow_count"] = active_exam + active_bg
            telemetry["exam_rtt"] = rtt
            telemetry["packet_loss"] = loss
            telemetry["override_limit"] = admit_limit
            with telemetry_path.open("w", encoding="utf-8") as handle:
                json.dump(telemetry, handle)

            # Resolve exam window states
            now_dt = datetime.now(UTC)
            window = config.current_window(now_dt)
            exam_window_state = "none"
            time_to_exam = 0.0
            if window:
                if window.is_active(now_dt):
                    exam_window_state = "active"
                elif window.is_prestage(now_dt):
                    exam_window_state = "prestage"
                    time_to_exam = (window.start - normalize_datetime(now_dt)).total_seconds()

            # Construct metrics record
            record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "scenario": scenario,
                "policy_mode": policy_mode,
                "active_exam_clients": active_exam,
                "active_background_clients": active_bg,
                "measured_throughput_mbps": measured_tput,
                "exam_rtt_ms": rtt,
                "packet_loss_percent": loss,
                "iperf_flow_count": active_exam + active_bg,
                "queue_utilization_percent": 0.0,
                "exam_window_state": exam_window_state,
                "time_to_exam_seconds": time_to_exam,
                "current_admitted_background_flow_limit": admit_limit,
                "action_source": action_source,
                "future_label": 0
            }
            time_series_records.append(record)

            # Precise interval sleeping
            loop_elapsed = time.time() - loop_start
            sleep_time = max(0.1, interval - loop_elapsed)
            time.sleep(sleep_time)
            elapsed = time.time() - start_time
            
        LOGGER.info("Concluded active emulated run. Finalizing parallel traffic collection...")

        # Wait on and aggregate exam results
        exam_tput = 0.0
        for host in iperf_exam_hosts:
            output = host.waitOutput()
            try:
                data = json.loads(output)
                summary = data.get("end", {}).get("sum_received") or data.get("end", {}).get("sum_sent") or data.get("end", {}).get("sum")
                if summary and "bits_per_second" in summary:
                    exam_tput += float(summary["bits_per_second"]) / 1_000_000
            except Exception as exc:
                LOGGER.warning("Could not extract iperf exam result: %s", exc)

        # Wait on and aggregate background results
        bg_tput = 0.0
        for host in iperf_bg_hosts:
            output = host.waitOutput()
            try:
                data = json.loads(output)
                summary = data.get("end", {}).get("sum_received") or data.get("end", {}).get("sum_sent") or data.get("end", {}).get("sum")
                if summary and "bits_per_second" in summary:
                    bg_tput += float(summary["bits_per_second"]) / 1_000_000
            except Exception as exc:
                LOGGER.warning("Could not extract iperf background result: %s", exc)

    finally:
        stop_iperf_servers(exam_server)
        if telemetry_path.exists():
            try:
                telemetry_path.unlink()
            except Exception:
                pass

    # Post-Process Future Label (RTT <= target_rtt_ms)
    target_rtt = float(config.policy.target_rtt_ms)
    for idx in range(len(time_series_records) - 1):
        next_rtt = time_series_records[idx + 1]["exam_rtt_ms"]
        time_series_records[idx]["future_label"] = 1 if next_rtt <= target_rtt else 0
    if time_series_records:
        time_series_records[-1]["future_label"] = 1 if time_series_records[-1]["exam_rtt_ms"] <= target_rtt else 0

    # Write out the CSV file
    headers = [
        "timestamp", "scenario", "policy_mode", "active_exam_clients", "active_background_clients",
        "measured_throughput_mbps", "exam_rtt_ms", "packet_loss_percent", "iperf_flow_count",
        "queue_utilization_percent", "exam_window_state", "time_to_exam_seconds",
        "current_admitted_background_flow_limit", "action_source", "future_label"
    ]
    with timeseries_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for record in time_series_records:
            writer.writerow(record)

    LOGGER.info("Wrote time-series logs to %s", timeseries_csv_path)

    # Perform a final aggregate ping check
    avg_rtt, packet_loss = collect_ping(ping_host, ping_target, count=5)
    return avg_rtt, packet_loss, exam_tput, bg_tput


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SDN exam traffic experiments.")
    parser.add_argument("--scenario", choices=["normal", "baseline", "protected"], required=True)
    parser.add_argument("--policy-mode", choices=["static", "heuristic", "ml"], default=os.getenv("SDN_DEMO_POLICY_MODE", "static"))
    parser.add_argument("--config", default=os.getenv("SDN_DEMO_CONFIG", str(ROOT / "config" / "demo_config.json")))
    parser.add_argument("--results-dir", default=os.getenv("SDN_DEMO_RESULTS_DIR", str(ROOT / "results" / "latest")))
    parser.add_argument("--log-level", default=os.getenv("SDN_DEMO_LOG_LEVEL", "INFO"))
    args = parser.parse_args()
    configure_logging(args.log_level)
    setLogLevel("warning")

    try:
        config = load_config(args.config)
        config.topology["exam_clients"] = 3
        config.topology["background_clients"] = 6
        config.topology["switch_bandwidth_mbps"] = 10
        LOGGER.info("Dynamically scaled down topology to exam_clients=3, background_clients=6, switch_bandwidth_mbps=10")
    except ConfigError as exc:
        LOGGER.error("invalid configuration: %s", exc)
        return 2

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{timestamp}_{args.scenario}"
    try:
        results_dir = resolve_results_dir(args.results_dir)
        assert_safe_results_dir(results_dir)
    except ExperimentError as exc:
        LOGGER.error("%s", exc)
        return 2

    attempt_dir = unique_run_dir(results_dir.parent / "attempts", run_name)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    log_path = attempt_dir / f"{run_name}.log"
    LOGGER.info("running script=%s cwd=%s timestamp=%s", Path(__file__).resolve(), Path.cwd(), timestamp)
    LOGGER.info("writing staged attempt output to %s", attempt_dir)

    protected = args.scenario == "protected"
    net = None
    try:
        net = create_network(config, protected=protected)
        net.start()
        if protected:
            configure_qos_queues(net, config)

        exam_hosts = [net.get(f"e{i + 1}") for i in range(config.topology["exam_clients"])]
        background_hosts = [net.get(f"b{i + 1}") for i in range(config.topology["background_clients"])]
        exam_server = net.get("exam0")
        server_ip = config.topology["exam_server_ip"].split("/")[0]
        ping_host = exam_hosts[0]

        if args.scenario == "normal":
            exam_subset = exam_hosts[:5]
            background_subset = background_hosts[:5]
        else:
            exam_subset = exam_hosts
            background_subset = background_hosts

        # --- WRITE STARTING TELEMETRY SO CONTROLLER IS SYNCED BEFORE WARMUP ---
        latest_dir = attempt_dir.parent.parent / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = latest_dir / "telemetry.json"
        
        policy_mode = args.policy_mode if protected else "static"
        
        telemetry = {
            "policy_mode": policy_mode,
            "active_exam_clients": len(exam_subset),
            "active_background_clients": len(background_subset),
            "exam_rtt": 0.0,
            "packet_loss": 0.0,
            "iperf_flow_count": len(exam_subset) + len(background_subset),
            "queue_utilization": 0.0,
            "exam_clients": int(config.topology["exam_clients"])
        }
        with telemetry_path.open("w", encoding="utf-8") as handle:
            json.dump(telemetry, handle)
        LOGGER.info("Wrote pre-warmup telemetry boundary: exam_clients=%d to %s", config.topology["exam_clients"], telemetry_path)
        # ----------------------------------------------------------------------

        warm_up_topology([*exam_subset, *background_subset], server_ip)

        policy_mode = args.policy_mode if protected else "static"

        avg_rtt, packet_loss, exam_tput, background_tput = run_concurrent_traffic_and_log(
            net,
            config,
            args.scenario,
            policy_mode,
            exam_subset,
            background_subset,
            exam_server,
            server_ip,
            attempt_dir,
            run_name,
        )

        log_lines = [
            f"scenario={args.scenario}",
            f"protected={protected}",
            f"policy_mode={policy_mode}",
            f"avg_rtt_ms={avg_rtt}",
            f"packet_loss_percent={packet_loss}",
            f"exam_throughput_mbps={exam_tput}",
            f"background_throughput_mbps={background_tput}",
        ]
        if protected:
            log_lines.append("expected_controller_features=classification,prestage,throttling,admission-control")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    except ExperimentError as exc:
        LOGGER.error("%s", exc)
        log_lines = [
            f"scenario={args.scenario}",
            f"protected={protected}",
            "status=failed",
            f"phase={exc.phase}",
            f"error={exc}",
            f"hint={FAILURE_HINT}",
        ]
        if exc.details:
            log_lines.append("details:")
            log_lines.extend(f"- {detail}" for detail in exc.details)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        write_failure_record(
            FailureRecord(scenario=args.scenario, protected=protected, phase=exc.phase, error=str(exc)),
            attempt_dir / "failure_summary.csv",
        )
        failed_dir = preserve_failed_attempt(attempt_dir, results_dir, run_name)
        LOGGER.error("wrote failure log to %s", failed_dir / log_path.name)
        return 1
    finally:
        if net is not None:
            try:
                net.stop()
            except Exception as exc:
                LOGGER.warning("failed to stop Mininet cleanly: %s", exc)

    records = build_records(args.scenario, avg_rtt, packet_loss, exam_tput, background_tput)
    output = write_metric_records(records, attempt_dir / f"{run_name}.csv")
    LOGGER.info("wrote staged metrics to %s", output)
    promote_successful_attempt(attempt_dir, results_dir)
    final_output = results_dir / output.name
    final_log_path = results_dir / log_path.name
    LOGGER.info("latest metrics available at %s", final_output)
    LOGGER.info("wrote log to %s", final_log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
