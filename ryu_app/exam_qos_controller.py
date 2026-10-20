from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdn_exam_demo.config import load_config
from sdn_exam_demo.policy import ExamBandwidthPolicy, FlowSpec, PolicyMode

try:
    from os_ken.base import app_manager
    from os_ken.controller import ofp_event
    from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
    from os_ken.lib.packet import ethernet, ipv4, packet, tcp, udp
    from os_ken.ofproto import ofproto_v1_3

    ControllerAppBase = app_manager.OSKenApp
except ImportError:  # pragma: no cover - runtime dependency in WSL/Ubuntu
    try:
        from ryu.base import app_manager
        from ryu.controller import ofp_event
        from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
        from ryu.lib.packet import ethernet, ipv4, packet, tcp, udp
        from ryu.ofproto import ofproto_v1_3

        ControllerAppBase = app_manager.RyuApp
    except ImportError as fallback_exc:  # pragma: no cover - runtime dependency in WSL/Ubuntu
        raise SystemExit(
            "OS-Ken or Ryu is required to run the controller. On Ubuntu/WSL run "
            "`sudo apt-get install python3-os-ken` and start with "
            "`osken-manager ryu_app/exam_qos_controller.py`."
        ) from fallback_exc

DEFAULT_DEMO_CONFIG = str(ROOT / "config" / "demo_config.json")


class ExamQoSController(ControllerAppBase):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    IDLE_TIMEOUT_SECONDS = 60
    HARD_TIMEOUT_SECONDS = 600

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config_file = kwargs.get("demo_config") or os.getenv("SDN_DEMO_CONFIG") or DEFAULT_DEMO_CONFIG
        self.demo_config = load_config(config_file)
        
        mode_str = os.getenv("SDN_DEMO_POLICY_MODE", "static").lower()
        if mode_str == "heuristic":
            mode = PolicyMode.HEURISTIC
        elif mode_str == "ml":
            mode = PolicyMode.ML
        else:
            mode = PolicyMode.STATIC
            
        self.logger.info("Initializing controller with policy mode: %s", mode)
        self.policy = ExamBandwidthPolicy(self.demo_config, mode=mode)
        self.mac_to_port: dict[int, dict[str, int]] = {}
        self.active_flow_keys: set[tuple[str, str, int | None, int | None, str]] = set()

    def _read_telemetry(self) -> dict[str, Any]:
        path = os.getenv("SDN_DEMO_TELEMETRY_PATH") or os.path.join(
            os.path.dirname(DEFAULT_DEMO_CONFIG), "..", "results", "latest", "telemetry.json"
        )
        resolved_path = Path(path).resolve()
        if resolved_path.exists():
            try:
                with open(resolved_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, dict):
                        if "policy_mode" in data:
                            mode_str = data["policy_mode"].lower()
                            if mode_str == "heuristic" and self.policy.mode != PolicyMode.HEURISTIC:
                                self.policy.mode = PolicyMode.HEURISTIC
                                self.logger.info("Dynamically switched policy mode to: HEURISTIC")
                            elif mode_str == "ml" and self.policy.mode != PolicyMode.ML:
                                self.policy.mode = PolicyMode.ML
                                self.logger.info("Dynamically switched policy mode to: ML")
                            elif mode_str == "static" and self.policy.mode != PolicyMode.STATIC:
                                self.policy.mode = PolicyMode.STATIC
                                self.logger.info("Dynamically switched policy mode to: STATIC")
                        if "exam_clients" in data:
                            old_exam_clients = self.demo_config.topology.get("exam_clients", 0)
                            new_exam_clients = int(data["exam_clients"])
                            if old_exam_clients != new_exam_clients:
                                self.demo_config.topology["exam_clients"] = new_exam_clients
                                self.logger.info("Dynamically updated topology.exam_clients boundary in config to: %d", new_exam_clients)
                    return data
            except Exception as exc:
                self.logger.warning("Failed to parse telemetry JSON from %s: %s", resolved_path, exc)
        return {}

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Reset active flow keys, policy counters, and established exam flows on reconnect/new switch features
        self.active_flow_keys.clear()
        self.policy.reset_counters()
        self.policy.established_exam_flows.clear()
        self.logger.info("New switch connection: cleared stale active flow keys and reset policy state.")

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, 0, match, actions, idle_timeout=0, hard_timeout=0)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None or eth.ethertype == 0x88CC:
            return

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][eth.src] = in_port
        out_port = self.mac_to_port[dpid].get(eth.dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt is not None:
            tcp_pkt = pkt.get_protocol(tcp.tcp)
            udp_pkt = pkt.get_protocol(udp.udp)
            flow = FlowSpec(
                src_ip=ip_pkt.src,
                dst_ip=ip_pkt.dst,
                protocol="tcp" if tcp_pkt else "udp" if udp_pkt else "ip",
                src_port=(tcp_pkt.src_port if tcp_pkt else udp_pkt.src_port if udp_pkt else None),
                dst_port=(tcp_pkt.dst_port if tcp_pkt else udp_pkt.dst_port if udp_pkt else None),
            )
            flow_key = self.policy.flow_key(flow)
            is_new_flow = flow_key not in self.active_flow_keys
            if is_new_flow:
                self.active_flow_keys.add(flow_key)
            measured_load_mbps = self._estimate_load_mbps()
            telemetry = self._read_telemetry()
            
            # Enrich telemetry with active counts and flow count if not present
            if "active_exam_clients" not in telemetry or "active_background_clients" not in telemetry:
                exam_flows = 0
                bg_flows = 0
                for fk in self.active_flow_keys:
                    fs = FlowSpec(src_ip=fk[0], dst_ip=fk[1], protocol=fk[4], src_port=fk[2], dst_port=fk[3])
                    if self.policy.classify(fs) == "exam":
                        exam_flows += 1
                    else:
                        bg_flows += 1
                telemetry["active_exam_clients"] = exam_flows
                telemetry["active_background_clients"] = bg_flows
            if "iperf_flow_count" not in telemetry:
                telemetry["iperf_flow_count"] = len(self.active_flow_keys)
                
            start_time = time.perf_counter()
            decision = self.policy.evaluate(
                flow,
                datetime.now(UTC),
                measured_load_mbps=measured_load_mbps,
                is_new_flow=is_new_flow,
                telemetry=telemetry,
            )
            decision_latency_ms = (time.perf_counter() - start_time) * 1000.0
            
            self.logger.info(
                "flow src=%s dst=%s class=%s queue=%s allow=%s est_load_mbps=%.2f limit=%d reasons=%s latency_ms=%.3f",
                flow.src_ip,
                flow.dst_ip,
                decision.traffic_class,
                decision.queue_id,
                decision.allow_new_flow,
                measured_load_mbps,
                decision.admitted_limit,
                ",".join(decision.reasons) or "none",
                decision_latency_ms,
            )
            if decision.allow_new_flow and out_port != ofproto.OFPP_FLOOD:
                actions = [parser.OFPActionSetQueue(decision.queue_id), parser.OFPActionOutput(out_port)]
                match = self._build_ip_match(parser, in_port, flow)
                self._add_flow(datapath, 10 if decision.traffic_class == "exam" else 5, match, actions)
            elif not decision.allow_new_flow:
                self.active_flow_keys.discard(flow_key)
                drop_match = self._build_ip_match(parser, in_port, flow)
                self._add_flow(datapath, 20, drop_match, [])
                return

        data = None if msg.buffer_id != ofproto.OFP_NO_BUFFER else msg.data
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        )
        datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def flow_removed_handler(self, ev):
        match = ev.msg.match
        src_ip = match.get("ipv4_src")
        dst_ip = match.get("ipv4_dst")
        if not src_ip or not dst_ip:
            return
        protocol = "ip"
        src_port = None
        dst_port = None
        if match.get("ip_proto") == 6:
            protocol = "tcp"
            src_port = match.get("tcp_src")
            dst_port = match.get("tcp_dst")
        elif match.get("ip_proto") == 17:
            protocol = "udp"
            src_port = match.get("udp_src")
            dst_port = match.get("udp_dst")
        self.active_flow_keys.discard((src_ip, dst_ip, src_port, dst_port, protocol))

    def _add_flow(
        self,
        datapath,
        priority,
        match,
        actions,
        buffer_id=None,
        idle_timeout: int | None = None,
        hard_timeout: int | None = None,
    ):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        kwargs = {
            "datapath": datapath,
            "priority": priority,
            "match": match,
            "instructions": inst,
            "idle_timeout": self.IDLE_TIMEOUT_SECONDS if idle_timeout is None else idle_timeout,
            "hard_timeout": self.HARD_TIMEOUT_SECONDS if hard_timeout is None else hard_timeout,
            "flags": ofproto.OFPFF_SEND_FLOW_REM,
        }
        if buffer_id is not None:
            kwargs["buffer_id"] = buffer_id
        mod = parser.OFPFlowMod(**kwargs)
        datapath.send_msg(mod)

    @staticmethod
    def _build_ip_match(parser, in_port, flow: FlowSpec):
        kwargs = {
            "in_port": in_port,
            "eth_type": 0x0800,
            "ipv4_src": flow.src_ip,
            "ipv4_dst": flow.dst_ip,
        }
        if flow.protocol == "tcp":
            kwargs["ip_proto"] = 6
            if flow.src_port is not None:
                kwargs["tcp_src"] = flow.src_port
            if flow.dst_port is not None:
                kwargs["tcp_dst"] = flow.dst_port
        elif flow.protocol == "udp":
            kwargs["ip_proto"] = 17
            if flow.src_port is not None:
                kwargs["udp_src"] = flow.src_port
            if flow.dst_port is not None:
                kwargs["udp_dst"] = flow.dst_port
        return parser.OFPMatch(**kwargs)

    def _estimate_load_mbps(self) -> float:
        exam_flows = 0
        background_flows = 0
        for fk in self.active_flow_keys:
            fs = FlowSpec(src_ip=fk[0], dst_ip=fk[1], protocol=fk[4], src_port=fk[2], dst_port=fk[3])
            if self.policy.classify(fs) == "exam":
                exam_flows += 1
            else:
                background_flows += 1
        return (
            exam_flows * float(self.demo_config.traffic["exam_flow_bandwidth_mbps"])
            + background_flows * float(self.demo_config.traffic["background_flow_bandwidth_mbps"])
        )
