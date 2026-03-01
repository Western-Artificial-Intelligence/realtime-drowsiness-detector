"""
Unit tests for fusion.combine() risk priority rules.
"""

import pytest
from src.wai.fusion import combine, RISK_OK, RISK_MONITOR, RISK_WARNING, RISK_CRITICAL


def test_critical_when_drowsy_alert_active():
    """Drowsiness alert wins over distraction."""
    out = combine(
        drowsy_metrics={"is_closed": True, "closure_duration": 2.0, "ear": 0.1},
        drowsy_alert={"alert_active": True, "reason": "Eyes closed 2.0s"},
        distraction_state={"triggered": True, "ratio": 0.9, "mean_prob": 0.85},
    )
    assert out["risk"] == RISK_CRITICAL
    assert "Eyes closed" in out["reason"]


def test_warning_when_distraction_triggered_no_drowsy_alert():
    """Distraction triggered → WARNING when no drowsy alert."""
    out = combine(
        drowsy_metrics={"is_closed": False, "closure_duration": 0.0},
        drowsy_alert={"alert_active": False, "reason": None},
        distraction_state={"triggered": True, "ratio": 0.8, "mean_prob": 0.7},
    )
    assert out["risk"] == RISK_WARNING
    assert "Distracted" in out["reason"]


def test_monitor_when_eyes_closed_briefly():
    """Eyes closed but not yet alert → MONITOR."""
    out = combine(
        drowsy_metrics={"is_closed": True, "closure_duration": 0.5},
        drowsy_alert={"alert_active": False, "reason": None},
        distraction_state={"triggered": False, "ratio": 0.2, "mean_prob": 0.3},
    )
    assert out["risk"] == RISK_MONITOR
    assert "Eyes closed" in out["reason"]


def test_monitor_when_high_distraction_mean_prob():
    """High mean_prob but not triggered → MONITOR."""
    out = combine(
        drowsy_metrics={"is_closed": False, "closure_duration": 0.0},
        drowsy_alert={"alert_active": False, "reason": None},
        distraction_state={"triggered": False, "ratio": 0.5, "mean_prob": 0.6},
    )
    assert out["risk"] == RISK_MONITOR
    assert "Distraction" in out["reason"]


def test_ok_when_all_low():
    """No alerts, not closed, low distraction → OK."""
    out = combine(
        drowsy_metrics={"is_closed": False, "closure_duration": 0.0},
        drowsy_alert={"alert_active": False, "reason": None},
        distraction_state={"triggered": False, "ratio": 0.1, "mean_prob": 0.2},
    )
    assert out["risk"] == RISK_OK
    assert out["reason"] == ""
