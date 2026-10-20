from datetime import datetime
from pathlib import Path

from sdn_exam_demo.config import load_config
from sdn_exam_demo.policy import ExamBandwidthPolicy, FlowSpec


def build_policy() -> ExamBandwidthPolicy:
    return ExamBandwidthPolicy(load_config(Path("config/demo_config.json")))


def test_exam_flow_gets_priority_queue():
    policy = build_policy()
    flow = FlowSpec(src_ip="10.0.0.10", dst_ip="10.0.0.200", protocol="tcp", src_port=50000, dst_port=443)
    decision = policy.evaluate(flow, datetime(2026, 4, 25, 9, 45, 0), measured_load_mbps=10)
    assert decision.traffic_class == "exam"
    assert decision.queue_id == 1
    assert "calendar-prestage" in decision.reasons


def test_background_flow_is_rate_limited_under_congestion():
    policy = build_policy()
    flow = FlowSpec(src_ip="10.0.0.30", dst_ip="10.0.0.99", protocol="udp", src_port=4000, dst_port=4001)
    decision = policy.evaluate(flow, datetime(2026, 4, 25, 10, 10, 0), measured_load_mbps=90)
    assert decision.traffic_class == "background"
    assert decision.rate_limit_percent == 10
    assert decision.allow_new_flow is True


def test_repeated_background_packet_does_not_consume_new_flow_budget():
    policy = build_policy()
    now = datetime(2026, 4, 25, 10, 10, 0)
    flow = FlowSpec(src_ip="10.0.0.30", dst_ip="10.0.0.99", protocol="udp", src_port=4000, dst_port=4001)

    first = policy.evaluate(flow, now, measured_load_mbps=90, is_new_flow=True)
    repeated = policy.evaluate(flow, now, measured_load_mbps=90, is_new_flow=False)

    assert first.allow_new_flow is True
    assert repeated.allow_new_flow is True
    assert policy.background_new_flows == 1


def test_admission_control_blocks_excess_background_flows():
    policy = build_policy()
    now = datetime(2026, 4, 25, 10, 10, 0)
    limit = policy.config.policy.max_background_new_flows
    for i in range(limit):
        flow = FlowSpec(src_ip=f"10.0.0.{100 + i}", dst_ip="10.0.0.99", protocol="tcp", src_port=1000 + i, dst_port=80)
        policy.evaluate(flow, now, measured_load_mbps=90)

    blocked = policy.evaluate(
        FlowSpec(src_ip="10.0.0.250", dst_ip="10.0.0.99", protocol="tcp", src_port=2500, dst_port=80),
        now,
        measured_load_mbps=90,
    )
    assert blocked.allow_new_flow is False
    assert "admission-control" in blocked.reasons


def test_heuristic_mode_bounds_admitted_limit():
    from sdn_exam_demo.policy import PolicyMode
    policy = ExamBandwidthPolicy(load_config(Path("config/demo_config.json")), mode=PolicyMode.HEURISTIC)
    now = datetime(2026, 4, 25, 10, 10, 0)
    flow = FlowSpec(src_ip="10.0.0.30", dst_ip="10.0.0.99", protocol="udp", src_port=4000, dst_port=4001)

    # Congested, with bad telemetry (high RTT) should clamp limit to min_background_new_flows
    telemetry = {"exam_rtt": 100.0, "packet_loss": 5.0}
    decision = policy.evaluate(flow, now, measured_load_mbps=90, telemetry=telemetry)
    assert decision.admitted_limit == policy.config.policy.min_background_new_flows
    assert "mode-heuristic" in decision.reasons

    # Low load, normal telemetry should result in a higher limit
    normal_now = datetime(2026, 4, 25, 9, 0, 0) # Outside exam window
    normal_telemetry = {"exam_rtt": 10.0, "packet_loss": 0.0}
    decision_normal = policy.evaluate(flow, normal_now, measured_load_mbps=1, telemetry=normal_telemetry)
    assert decision_normal.admitted_limit > policy.config.policy.min_background_new_flows


def test_ml_mode_falls_back_without_model(monkeypatch):
    from sdn_exam_demo.policy import PolicyMode
    monkeypatch.setenv("SDN_DEMO_MODEL_PATH", "non_existent_model.joblib")
    policy = ExamBandwidthPolicy(load_config(Path("config/demo_config.json")), mode=PolicyMode.ML)
    now = datetime(2026, 4, 25, 10, 10, 0)
    flow = FlowSpec(src_ip="10.0.0.30", dst_ip="10.0.0.99", protocol="udp", src_port=4000, dst_port=4001)

    # Under no model loaded, it should fall back to heuristic
    decision = policy.evaluate(flow, now, measured_load_mbps=95, telemetry={"exam_rtt": 100.0})
    assert decision.admitted_limit == 1
    assert "mode-ml" in decision.reasons


def test_exam_flow_bypasses_blocked_background_budget():
    policy = build_policy()
    now = datetime(2026, 4, 25, 10, 10, 0)  # Midterm active window
    limit = policy.config.policy.max_background_new_flows

    # 1. Saturate the background budget completely so background is blocked
    for i in range(limit):
        bg_flow = FlowSpec(src_ip=f"10.0.0.{100 + i}", dst_ip="10.0.0.99", protocol="tcp", src_port=1000 + i, dst_port=80)
        policy.evaluate(bg_flow, now, measured_load_mbps=90)

    # Verify next background flow is indeed blocked
    bg_blocked = policy.evaluate(
        FlowSpec(src_ip="10.0.0.250", dst_ip="10.0.0.99", protocol="tcp", src_port=2500, dst_port=80),
        now,
        measured_load_mbps=90,
    )
    assert bg_blocked.allow_new_flow is False
    assert "admission-control" in bg_blocked.reasons

    # 2. Interleave an exam flow packet now!
    exam_flow = FlowSpec(src_ip="10.0.0.10", dst_ip="10.0.0.200", protocol="tcp", src_port=50000, dst_port=443)
    exam_decision = policy.evaluate(exam_flow, now, measured_load_mbps=90)

    # Verify that the exam flow is allowed and correctly put in the priority queue
    assert exam_decision.allow_new_flow is True
    assert exam_decision.traffic_class == "exam"
    assert exam_decision.queue_id == policy.config.policy.priority_queue_id
    assert "exam-destination-ip" in exam_decision.reasons


def test_ml_mode_with_actual_model(monkeypatch):
    import os
    from sdn_exam_demo.policy import PolicyMode
    # Force use of the actual serialized admitter_model.joblib
    actual_model_path = str(Path("src/sdn_exam_demo/models/admitter_model.joblib").resolve())
    monkeypatch.setenv("SDN_DEMO_MODEL_PATH", actual_model_path)
    
    policy = ExamBandwidthPolicy(load_config(Path("config/demo_config.json")), mode=PolicyMode.ML)
    now = datetime(2026, 4, 25, 10, 10, 0) # Midterm active window
    flow = FlowSpec(src_ip="10.0.0.30", dst_ip="10.0.0.99", protocol="udp", src_port=4000, dst_port=4001)

    # Congested state under ML mode
    telemetry = {
        "active_exam_clients": 3.0,
        "active_background_clients": 6.0,
        "exam_rtt": 0.5,
        "packet_loss": 0.0,
        "iperf_flow_count": 9.0,
        "queue_utilization": 0.0
    }
    
    # 1. Congested state (measured load = 90 Mbps > congestion threshold 10 Mbps)
    decision = policy.evaluate(flow, now, measured_load_mbps=90, telemetry=telemetry)
    assert decision.traffic_class == "background"
    assert "mode-ml" in decision.reasons
    assert policy.config.policy.min_background_new_flows <= decision.admitted_limit <= policy.config.policy.max_background_new_flows


def test_ml_mode_with_mocked_model():
    from sdn_exam_demo.policy import PolicyMode
    
    policy = ExamBandwidthPolicy(load_config(Path("config/demo_config.json")), mode=PolicyMode.ML)
    
    # Mock self.ml_model to return specific predictions
    class MockModel:
        def predict(self, df):
            # If RTT is high, recommend safety floor (1.0), otherwise max limit (4.0)
            if df.loc[0, "exam_rtt_ms"] > 10.0:
                return [1.0]
            return [4.0]
            
    policy.ml_model = MockModel()
    policy._ml_model_loaded = True
    
    now = datetime(2026, 4, 25, 10, 10, 0)
    flow = FlowSpec(src_ip="10.0.0.30", dst_ip="10.0.0.99", protocol="udp", src_port=4000, dst_port=4001)

    # 1. Congested state with high RTT -> should predict 1.0 (clamped to safety floor)
    telemetry_congested = {"exam_rtt": 15.0}
    decision = policy.evaluate(flow, now, measured_load_mbps=90, telemetry=telemetry_congested)
    assert decision.admitted_limit == 1
    
    # 2. Stable state with low RTT -> should predict 4.0 (normal max limit)
    telemetry_stable = {"exam_rtt": 0.5}
    decision_stable = policy.evaluate(flow, now, measured_load_mbps=90, telemetry=telemetry_stable)
    assert decision_stable.admitted_limit == 4



