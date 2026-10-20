from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdn_exam_demo.config import load_config

try:
    from mininet.cli import CLI
    from mininet.link import Link
    from mininet.log import setLogLevel
    from mininet.net import Mininet
    from mininet.node import OVSSwitch, RemoteController
    from mininet.topo import Topo
except ImportError as exc:  # pragma: no cover - runtime dependency in WSL/Ubuntu
    raise SystemExit(
        "Mininet is required to run the topology. Install Ubuntu/WSL dependencies and run this file from Ubuntu."
    ) from exc


class ExamTrafficTopo(Topo):
    def build(self, exam_clients: int, background_clients: int, exam_server_ip: str):
        switch = self.addSwitch("s1")
        server = self.addHost("exam0", ip=exam_server_ip)
        self.addLink(server, switch, cls=Link)

        for index in range(exam_clients):
            host = self.addHost(f"e{index + 1}", ip=f"10.0.0.{index + 1}/24")
            self.addLink(host, switch, cls=Link)

        for index in range(background_clients):
            base_ip = index + exam_clients + 1
            host = self.addHost(f"b{index + 1}", ip=f"10.0.0.{base_ip}/24")
            self.addLink(host, switch, cls=Link)


def build_topology(config):
    return ExamTrafficTopo(
        exam_clients=config.topology["exam_clients"],
        background_clients=config.topology["background_clients"],
        exam_server_ip=config.topology["exam_server_ip"],
    )


def create_network(config, protected: bool):
    # Dynamic self-clean: Purge lingering virtual interfaces from the kernel before Mininet starts
    try:
        import subprocess
        out = subprocess.check_output("ip link show", shell=True).decode()
        for line in out.splitlines():
            if "-eth" in line or "s1-eth" in line:
                parts = line.split()
                if len(parts) >= 2:
                    intf = parts[1].split("@")[0].strip(":")
                    subprocess.run(f"ip link delete {intf} 2>/dev/null", shell=True)
    except Exception:
        pass

    topo = build_topology(config)
    switch_factory = OVSSwitch
    net = Mininet(
        topo=topo,
        controller=None,
        autoSetMacs=True,
        waitConnected=True,
        switch=switch_factory,
        link=Link,
    )
    net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6633)
    return net


def configure_qos_queues(net, config) -> None:
    """Attach Linux HTB queues to switch ports for protected experiments."""
    switch = net.get("s1")
    max_rate = int(config.topology["switch_bandwidth_mbps"]) * 1_000_000
    floor_rate = max(1, int(max_rate * int(config.policy.minimum_service_floor_percent) / 100))
    priority_rate = max(floor_rate, int(max_rate * 0.75))
    best_effort_queue = int(config.policy.best_effort_queue_id)
    priority_queue = int(config.policy.priority_queue_id)

    for interface in switch.intfNames():
        if interface == "lo":
            continue
        switch.cmd(f"ovs-vsctl --if-exists clear port {interface} qos")
        switch.cmd(
            "ovs-vsctl "
            f"-- set port {interface} qos=@qos "
            f"-- --id=@qos create qos type=linux-htb other-config:max-rate={max_rate} "
            f"queues:{best_effort_queue}=@best queues:{priority_queue}=@priority "
            f"-- --id=@best create queue other-config:min-rate={floor_rate} "
            f"-- --id=@priority create queue other-config:min-rate={priority_rate}"
        )


def main():
    config = load_config(ROOT / "config" / "demo_config.json")
    net = create_network(config, protected=True)
    net.start()
    configure_qos_queues(net, config)
    print("Topology started. Use Mininet CLI for manual testing.")
    CLI(net)
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    main()
