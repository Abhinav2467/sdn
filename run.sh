#!/usr/bin/env bash
# SDN Exam Traffic Demo ML Mode Startup Wrapper
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Ensure virtualenv is active
if [ ! -d ".venv" ]; then
  echo "Error: Virtual environment (.venv) not found. Run scripts/setup_wsl.sh first!"
  exit 1
fi

echo "=========================================================="
echo "Starting SDN Exam Traffic Demo - ML-Assisted Mode"
echo "=========================================================="

# Clean up any stale Mininet/OVS states
echo "Cleaning up lingering networking interfaces..."
echo "123" | sudo -S mn -c
echo "123" | sudo -S service openvswitch-switch restart

# Start Ryu QoS Controller in ML mode in the background
echo "Launching Ryu Controller in ML Mode..."
export PYTHONPATH="$ROOT/.venv/lib/python3.12/site-packages"
export SDN_DEMO_POLICY_MODE=ml
export SDN_DEMO_CONFIG="$ROOT/config/demo_config.json"
export SDN_DEMO_MODEL_PATH="$ROOT/src/sdn_exam_demo/models/admitter_model.joblib"

osken-manager ryu_app/exam_qos_controller.py > controller_run.log 2>&1 &
CONTROLLER_PID=$!

# Wait for controller to initialize
sleep 3
if ! ps -p $CONTROLLER_PID > /dev/null; then
  echo "Error: Ryu controller failed to start. Check controller_run.log for details!"
  exit 1
fi
echo "Ryu Controller successfully initialized (PID: $CONTROLLER_PID)."

# Run Mininet emulated protected experiment
echo "Starting Mininet emulation & executing protected experiment..."
echo "123" | sudo -S .venv/bin/python scripts/run_experiment.py --scenario protected --policy-mode ml --config config/demo_config.json

# Clean up controller process on exit
echo "Terminating Ryu Controller..."
kill $CONTROLLER_PID || true
echo "123" | sudo -S mn -c

echo "=========================================================="
echo "Experiment execution complete. Results saved under results/latest/"
echo "=========================================================="
