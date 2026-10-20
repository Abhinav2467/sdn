from datetime import datetime
from pathlib import Path
from sdn_exam_demo.config import load_config
from sdn_exam_demo.policy.heuristic_policy import HeuristicPolicy

def test_heuristic_policy_normal_conditions():
    config = load_config(Path("config/demo_config.json"))
    policy = HeuristicPolicy(config)
    
    # Outside exam window (e.g. 9:00 AM on April 25), low load, clean telemetry
    now = datetime(2026, 4, 25, 9, 0, 0)
    telemetry = {"exam_rtt": 5.0, "packet_loss": 0.0}
    limit = policy.evaluate(measured_load_mbps=1.0, telemetry=telemetry, now=now)
    
    # Base limit is 4.0 (max_background_new_flows in demo_config.json)
    assert limit == 4

def test_heuristic_policy_bounds():
    config = load_config(Path("config/demo_config.json"))
    policy = HeuristicPolicy(config)
    
    now = datetime(2026, 4, 25, 9, 0, 0)
    
    # Extreme telemetry (high RTT) should clamp to 1.0 (min_background_new_flows)
    telemetry_bad = {"exam_rtt": 100.0, "packet_loss": 10.0}
    limit_bad = policy.evaluate(measured_load_mbps=100.0, telemetry=telemetry_bad, now=now)
    assert limit_bad == 1.0
    
    # Clean telemetry, but extremely low load should not exceed 4.0 (max_background_new_flows)
    telemetry_clean = {"exam_rtt": 1.0, "packet_loss": 0.0}
    limit_clean = policy.evaluate(measured_load_mbps=0.0, telemetry=telemetry_clean, now=now)
    assert limit_clean == 4.0

def test_heuristic_policy_exam_window():
    config = load_config(Path("config/demo_config.json"))
    policy = HeuristicPolicy(config)
    
    # During exam window (e.g. 10:10 AM on April 25), base limit is reduced by 60%
    now_active = datetime(2026, 4, 25, 10, 10, 0)
    telemetry = {"exam_rtt": 2.0, "packet_loss": 0.0}
    
    # Base = 4. During active: 4 * 0.4 = 1.6 (clamped to int -> 1)
    limit_active = policy.evaluate(measured_load_mbps=1.0, telemetry=telemetry, now=now_active)
    assert limit_active == 1

def test_heuristic_policy_prestage_window():
    config = load_config(Path("config/demo_config.json"))
    policy = HeuristicPolicy(config)
    
    # During prestage window (e.g. 9:45 AM on April 25), base limit is reduced by 30%
    now_prestage = datetime(2026, 4, 25, 9, 45, 0)
    telemetry = {"exam_rtt": 2.0, "packet_loss": 0.0}
    
    # Base = 4. During prestage: 4 * 0.7 = 2.8 -> int(2.8) = 2.
    limit_prestage = policy.evaluate(measured_load_mbps=1.0, telemetry=telemetry, now=now_prestage)
    assert limit_prestage == 2

def test_heuristic_policy_congestion_scaling():
    config = load_config(Path("config/demo_config.json"))
    policy = HeuristicPolicy(config)
    
    now = datetime(2026, 4, 25, 9, 0, 0)
    telemetry = {"exam_rtt": 2.0, "packet_loss": 0.0}
    
    # Under congestion threshold (congestion_threshold_mbps is 10.0 in config)
    # At 9 Mbps (ratio = 9/10 = 0.9 > 0.8), limit is reduced by 20%
    # 4 * 0.8 = 3.2 -> int -> 3
    limit_mod = policy.evaluate(measured_load_mbps=9.0, telemetry=telemetry, now=now)
    assert limit_mod == 3
    
    # At 12 Mbps (ratio = 12/10 = 1.2 >= 1.0), limit is reduced by 50%
    # 4 * 0.5 = 2.0 -> 2
    limit_congested = policy.evaluate(measured_load_mbps=12.0, telemetry=telemetry, now=now)
    assert limit_congested == 2
