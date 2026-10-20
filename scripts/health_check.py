#!/usr/bin/env python3
from __future__ import annotations


import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdn_exam_demo.environment import controller_checks, integration_checks, python_checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SDN demo preflight checks.")
    parser.add_argument("--mode", choices=["python", "controller", "integration"], default="python")
    parser.add_argument("--config", default=str(ROOT / "config" / "demo_config.json"))
    args = parser.parse_args()

    if args.mode == "python":
        results = python_checks(args.config)
    elif args.mode == "controller":
        results = controller_checks(args.config)
    else:
        results = integration_checks(args.config)

    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
