#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  git \
  iperf3 \
  openvswitch-switch \
  python3-os-ken \
  python3 \
  python3-pip \
  python3-venv \
  tcpdump \
  mininet

python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/health_check.py --mode python

if command -v osken-manager >/dev/null 2>&1; then
  osken-manager --version || true
fi

echo "WSL setup complete."
