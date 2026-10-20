from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, load_config


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_python(min_version: tuple[int, int] = (3, 12)) -> CheckResult:
    current = sys.version_info
    ok = (current.major, current.minor) >= min_version
    detail = f"{platform.python_version()} at {sys.executable}"
    return CheckResult("python", ok, detail)


def check_import(module_name: str) -> CheckResult:
    spec = importlib.util.find_spec(module_name)
    return CheckResult(f"import:{module_name}", spec is not None, "available" if spec else "missing")


def check_command(command: str) -> CheckResult:
    path = shutil.which(command)
    return CheckResult(f"command:{command}", path is not None, path or "missing")


def check_config(config_path: str | Path) -> CheckResult:
    try:
        load_config(config_path)
    except ConfigError as exc:
        return CheckResult("config", False, str(exc))
    return CheckResult("config", True, str(config_path))


def is_linux() -> bool:
    return platform.system().lower() == "linux"


def has_root_privileges() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def python_checks(config_path: str | Path) -> list[CheckResult]:
    return [
        check_python(),
        check_config(config_path),
        check_import("pandas"),
        check_import("matplotlib"),
    ]


def controller_checks(config_path: str | Path) -> list[CheckResult]:
    return python_checks(config_path) + [
        check_import("os_ken"),
        check_command("osken-manager"),
    ]


def integration_checks(config_path: str | Path) -> list[CheckResult]:
    linux = CheckResult("platform:linux", is_linux(), platform.system())
    root = CheckResult(
        "privileges:root",
        has_root_privileges(),
        "root" if has_root_privileges() else "run Mininet experiments with sudo",
    )
    return controller_checks(config_path) + [
        linux,
        root,
        check_command("mn"),
        check_command("ovs-vsctl"),
        check_command("iperf3"),
    ]
