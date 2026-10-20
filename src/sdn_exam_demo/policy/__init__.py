from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import os
from typing import Any

from ..config import DemoConfig, normalize_datetime
from .heuristic_policy import HeuristicPolicy

LOGGER = logging.getLogger("sdn_exam_demo.policy")


class PolicyMode(Enum):
    STATIC = "static"
    HEURISTIC = "heuristic"
    ML = "ml"


@dataclass(frozen=True)
class FlowSpec:
    src_ip: str
    dst_ip: str
    protocol: str
    src_port: int | None = None
    dst_port: int | None = None


@dataclass
class PolicyDecision:
    queue_id: int
    rate_limit_percent: int | None = None
    allow_new_flow: bool = True
    traffic_class: str = "background"
    reasons: list[str] = field(default_factory=list)
    admitted_limit: int = 0


class ExamBandwidthPolicy:
    def __init__(self, config: DemoConfig, mode: PolicyMode = PolicyMode.STATIC) -> None:
        self.config = config
        self.mode = mode
        self.established_exam_flows: set[tuple[str, str, int | None, int | None, str]] = set()
        self.background_new_flows = 0
        
        self.ml_model = None
        self._ml_model_loaded = False
        self._load_ml_model()
        self.heuristic_policy = HeuristicPolicy(config)

    def classify(self, flow: FlowSpec) -> str:
        # Check if either src or dst is the exam portal
        src_is_portal = flow.src_ip in self.config.exam_portal_ips
        dst_is_portal = flow.dst_ip in self.config.exam_portal_ips
        
        if not (src_is_portal or dst_is_portal):
            return "background"
            
        # The other IP must be the client
        client_ip = flow.dst_ip if src_is_portal else flow.src_ip
        
        try:
            # Check if the client IP is in the exam clients range (1 to exam_clients)
            parts = client_ip.split(".")
            if len(parts) == 4 and parts[0] == "10" and parts[1] == "0" and parts[2] == "0":
                last_octet = int(parts[3])
                if 1 <= last_octet <= int(self.config.topology["exam_clients"]):
                    return "exam"
        except Exception:
            pass
            
        return "background"

    def track_established_flow(self, flow: FlowSpec) -> None:
        if self.classify(flow) == "exam":
            self.established_exam_flows.add(self.flow_key(flow))

    def is_established_exam_flow(self, flow: FlowSpec) -> bool:
        return self.flow_key(flow) in self.established_exam_flows

    def _load_ml_model(self) -> None:
        if self._ml_model_loaded:
            return
        self._ml_model_loaded = True
        try:
            import joblib
            model_path = os.getenv("SDN_DEMO_MODEL_PATH") or os.path.join(
                os.path.dirname(__file__), "..", "models", "admitter_model.joblib"
            )
            if os.path.exists(model_path):
                self.ml_model = joblib.load(model_path)
                LOGGER.info("Successfully loaded ML model from %s", model_path)
            else:
                LOGGER.warning("ML model file not found at %s. Will fall back to heuristic.", model_path)
        except Exception as exc:
            LOGGER.error("Failed to load ML model: %s. Will fall back to heuristic.", exc)

    def evaluate_heuristic(
        self,
        measured_load_mbps: float,
        telemetry: dict[str, Any] | None,
        now: datetime,
    ) -> int:
        return self.heuristic_policy.evaluate(measured_load_mbps, telemetry, now)

    def evaluate_ml(
        self,
        measured_load_mbps: float,
        telemetry: dict[str, Any] | None,
        now: datetime,
    ) -> int:
        self._load_ml_model()
        
        if self.ml_model is None or telemetry is None:
            # Fall back to heuristic
            return self.evaluate_heuristic(measured_load_mbps, telemetry, now)
            
        try:
            # Build feature DataFrame matching the exact column names used in train_model.py.
            # Using a named DataFrame instead of a plain list eliminates the sklearn
            # UserWarning: "X does not have valid feature names".
            import pandas as pd

            window = self.config.current_window(now)
            exam_window_active = 1.0 if (window and window.is_active(now)) else 0.0
            exam_window_prestage = 1.0 if (window and window.is_prestage(now)) else 0.0

            time_to_exam = 0.0
            if window and window.is_prestage(now):
                time_to_exam = float((window.start - normalize_datetime(now)).total_seconds())

            # Column names must exactly match those used in train_model.py → prepare_features_and_targets()
            feature_df = pd.DataFrame([{
                "active_exam_clients":       float(telemetry.get("active_exam_clients", 0.0)),
                "active_background_clients": float(telemetry.get("active_background_clients", 0.0)),
                "measured_throughput_mbps":  float(measured_load_mbps),
                "exam_rtt_ms":               float(telemetry.get("exam_rtt", 0.0)),
                "packet_loss_percent":       float(telemetry.get("packet_loss", 0.0)),
                "iperf_flow_count":          float(telemetry.get("iperf_flow_count", 0.0)),
                "queue_utilization_percent": float(telemetry.get("queue_utilization", 0.0)),
                "exam_window_active":        exam_window_active,
                "exam_window_prestage":      exam_window_prestage,
                "time_to_exam_seconds":      time_to_exam,
            }])

            prediction = self.ml_model.predict(feature_df)[0]
            # Ensure prediction is a bounded integer
            limit = int(round(prediction))

            policy_cfg = self.config.policy
            return max(policy_cfg.min_background_new_flows, min(limit, policy_cfg.max_background_new_flows))
        except Exception as exc:
            LOGGER.error("Error executing ML model prediction: %s. Falling back to heuristic.", exc)
            return self.evaluate_heuristic(measured_load_mbps, telemetry, now)

    def evaluate(
        self,
        flow: FlowSpec,
        now: datetime,
        measured_load_mbps: float,
        is_new_flow: bool = True,
        telemetry: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        now = normalize_datetime(now)
        traffic_class = self.classify(flow)
        window = self.config.current_window(now)
        congested = measured_load_mbps >= self.config.policy.congestion_threshold_mbps
        decision = PolicyDecision(
            queue_id=self.config.policy.best_effort_queue_id,
            traffic_class=traffic_class,
        )

        if traffic_class == "exam":
            decision.queue_id = self.config.policy.priority_queue_id
            decision.reasons.append("exam-destination-ip")
            if window and window.is_prestage(now):
                decision.reasons.append("calendar-prestage")
            if window and (window.is_active(now) or window.is_prestage(now)):
                self.track_established_flow(flow)
            return decision

        if window and window.is_prestage(now):
            decision.reasons.append("prestage-window-active")

        # Resolve active background flows limit based on policy mode
        if self.mode == PolicyMode.STATIC:
            limit = self.config.policy.max_background_new_flows
            decision.reasons.append("mode-static")
        elif self.mode == PolicyMode.HEURISTIC:
            limit = self.evaluate_heuristic(measured_load_mbps, telemetry, now)
            decision.reasons.append("mode-heuristic")
        elif self.mode == PolicyMode.ML:
            limit = self.evaluate_ml(measured_load_mbps, telemetry, now)
            decision.reasons.append("mode-ml")
        else:
            limit = self.config.policy.max_background_new_flows
            decision.reasons.append("mode-fallback")

        decision.admitted_limit = limit

        if congested:
            decision.rate_limit_percent = self.config.policy.minimum_service_floor_percent
            decision.reasons.append("graceful-throttling")
            
        # Apply admission control using resolved limit (independent of congestion, or under congestion)
        # Typically, admission control applies when congested or when the limit is exceeded.
        # To match legacy behavior, we block new flows exceeding the calculated limit during congestion.
        if congested:
            if is_new_flow and self.background_new_flows >= limit:
                decision.allow_new_flow = False
                decision.reasons.append("admission-control")
            elif is_new_flow:
                self.background_new_flows += 1

        return decision

    def reset_counters(self) -> None:
        self.background_new_flows = 0

    @staticmethod
    def flow_key(flow: FlowSpec) -> tuple[str, str, int | None, int | None, str]:
        return (flow.src_ip, flow.dst_ip, flow.src_port, flow.dst_port, flow.protocol)

    @staticmethod
    def _flow_key(flow: FlowSpec) -> tuple[str, str, int | None, int | None, str]:
        return ExamBandwidthPolicy.flow_key(flow)
