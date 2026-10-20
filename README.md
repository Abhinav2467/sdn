# SDN Exam Traffic Demo

SDN Exam Traffic Demo is a Python project for simulating congestion control that protects online exam traffic during high-load network events. It uses a reusable pure-Python policy layer plus a Linux-only Mininet/Open vSwitch/OS-Ken integration path.

## Features

- Classifies exam traffic by destination portal IP.
- Pre-stages exam flows during scheduled exam windows.
- Prioritizes exam traffic with OpenFlow queue actions.
- Throttles and limits new background flows under congestion.
- Records experiment metrics as CSV and plots results.
- Provides preflight checks to avoid cascading Mininet/OVS/controller errors.
- Stages experiment output so failed runs do not overwrite the last successful results.

## Tech Stack

- Python 3.12 or newer for pure-Python code and tests.
- Ubuntu 24.04 / WSL2 for SDN integration.
- Mininet 2.3, Open vSwitch 3.x, OpenFlow 1.3.
- OS-Ken from Ubuntu package `python3-os-ken` for the controller runner.
- pandas and matplotlib for results analysis.
- pytest, Ruff, mypy, and pip-audit for quality checks.

## Architecture

```text
config/demo_config.json      Experiment topology, traffic, schedule, and policy config
src/sdn_exam_demo/           Shared config validation, policy, logging, results, checks
ryu_app/                     OS-Ken/Ryu-compatible OpenFlow controller app
mininet/                     Reproducible Mininet topology and QoS queue setup
scripts/                     Setup, health checks, experiment runner, plotting
tests/                       Pure-Python unit tests
```

The controller imports OS-Ken first and falls back to legacy Ryu only if available. On Ubuntu 24.04, install `python3-os-ken`; the current PyPI `os-ken` wheel provides libraries but not the `osken-manager` executable.

## Requirements

For pure development:

- Python 3.12+
- pip

For full experiments:

- Ubuntu 24.04 or WSL2 Ubuntu
- sudo access
- `python3-os-ken`
- `mininet`
- `openvswitch-switch`
- `iperf3`
- `tcpdump`

## Installation

Pure-Python setup:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/health_check.py --mode python
```

Ubuntu/WSL full setup:

```bash
bash scripts/setup_wsl.sh
. .venv/bin/activate
python scripts/health_check.py --mode controller
```

If the controller check fails because `osken-manager` is missing:

```bash
sudo apt-get update
sudo apt-get install -y python3-os-ken
```

## Environment Variables

```bash
SDN_DEMO_CONFIG=config/demo_config.json
SDN_DEMO_RESULTS_DIR=results/latest
SDN_DEMO_LOG_LEVEL=INFO
```

## Running

Start the controller in one Ubuntu/WSL terminal:

```bash
osken-manager ryu_app/exam_qos_controller.py
```

For a non-default JSON config path:

```bash
SDN_DEMO_CONFIG=config/demo_config.json osken-manager ryu_app/exam_qos_controller.py
```

Run an experiment in another Ubuntu/WSL terminal:

```bash
sudo .venv/bin/python scripts/run_experiment.py --scenario protected --config config/demo_config.json
```

Supported scenarios are `normal`, `baseline`, and `protected`.

By default, each run writes first to a timestamped attempt directory. Successful runs replace `results/latest`; failed runs are preserved under `results/failed/<timestamp>_<scenario>/` with a failure log and `failure_summary.csv`. This keeps the last successful CSV available for plotting and avoids misleading metrics from failed runs. The runner refuses unsafe output targets such as the repository root, home directory, filesystem root, or WSL drive roots.

Generate plots:

```bash
python scripts/plot_results.py --results-dir results/latest
```

## Testing And Quality

```bash
python -m pytest -q
ruff check .
mypy src scripts tests
python -m compileall -q src scripts tests ryu_app mininet
python scripts/health_check.py --mode python
python -m pip_audit -r requirements.txt
```

Full integration preflight on Ubuntu/WSL:

```bash
python scripts/health_check.py --mode integration
```

## Docker

```bash
docker build -t sdn-exam-demo .
docker run --rm sdn-exam-demo
```

Mininet/OVS experiments inside Docker require privileged networking and host kernel support. Prefer WSL2/Ubuntu for repeatable local experiments.

## Configuration

`config/demo_config.json` controls exam portal IPs, exam schedule, policy thresholds, topology, and synthetic traffic. Configuration is validated at startup. Invalid JSON, invalid IPs, invalid port ranges, missing fields, reversed schedules, and out-of-range percentages fail fast.

## Troubleshooting

Install missing controller runner:

```bash
sudo apt-get install -y python3-os-ken
```

Clean stale Mininet state:

```bash
sudo mn -c
sudo systemctl restart openvswitch-switch
```

If a run exits early with a topology reachability or iperf host failure, inspect the newest directory under `results/failed/`, clean Mininet state, confirm the controller is running, and retry:

```bash
sudo mn -c
sudo .venv/bin/python scripts/run_experiment.py --scenario protected --config config/demo_config.json
```

Check prerequisites:

```bash
python scripts/health_check.py --mode integration
```

## Production Recommendations

- Replace synthetic load estimation with switch stats or telemetry.
- Define real queue and meter policies for your OVS/SDN environment.
- Add controller process supervision with systemd or orchestration.
- Export logs and metrics to your monitoring platform.
- Test failover behavior when the controller is unavailable.

## Scripts

- `scripts/setup_wsl.sh`: installs Ubuntu packages, creates a system-site venv, installs Python tools, and runs a basic health check.
- `scripts/health_check.py`: verifies Python, config, controller, and integration prerequisites.
- `scripts/run_experiment.py`: creates the Mininet topology and writes CSV/log results.
- `scripts/plot_results.py`: creates metric plots from result CSV files.

## API Surface

- `sdn_exam_demo.config.load_config(path)`: loads and validates JSON config.
- `sdn_exam_demo.policy.ExamBandwidthPolicy.evaluate(...)`: returns queue/admission decisions for a flow.
- `sdn_exam_demo.results.write_metric_records(...)`: writes metrics to CSV.
- `sdn_exam_demo.environment.*_checks(...)`: returns structured preflight results.
