from __future__ import annotations
from datetime import datetime
from typing import Any
from sdn_exam_demo.config import DemoConfig

class HeuristicPolicy:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config

    def evaluate(
        self,
        measured_load_mbps: float,
        telemetry: dict[str, Any] | None,
        now: datetime,
    ) -> int:
        policy_cfg = self.config.policy
        window = self.config.current_window(now)
        
        # Base limit
        limit = policy_cfg.max_background_new_flows
        
        # 1. Check prestage / active exam window
        if window:
            if window.is_active(now):
                limit = int(limit * 0.4)  # Reduce by 60% during exam
            elif window.is_prestage(now):
                limit = int(limit * 0.7)  # Reduce by 30% during prestage
        
        # 2. Check telemetry RTT and Packet Loss
        if telemetry:
            rtt = telemetry.get("exam_rtt", 0.0)
            loss = telemetry.get("packet_loss", 0.0)
            
            # If targets are violated, clamp to minimum guardrail immediately
            if rtt >= policy_cfg.target_rtt_ms or loss >= policy_cfg.target_packet_loss_percent:
                return int(policy_cfg.min_background_new_flows)
            
            # Gradients if getting close to thresholds
            if rtt > policy_cfg.target_rtt_ms * 0.7:
                limit = int(limit * 0.6)
            if loss > policy_cfg.target_packet_loss_percent * 0.5:
                limit = int(limit * 0.5)
                
        # 3. Congestion scaling
        load_ratio = measured_load_mbps / policy_cfg.congestion_threshold_mbps
        if load_ratio >= 1.0:
            limit = int(limit * 0.5)
        elif load_ratio > 0.8:
            limit = int(limit * 0.8)
            
        # Ensure we are strictly bounded by min and max guardrails
        return int(max(policy_cfg.min_background_new_flows, min(limit, policy_cfg.max_background_new_flows)))
