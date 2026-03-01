"""
Minimal fusion of drowsiness and distraction into a single top-level risk state.

Used for demo: one combined risk level and reason for the HUD and alerts.
"""

from typing import TypedDict

__all__ = [
    "RISK_OK",
    "RISK_MONITOR",
    "RISK_WARNING",
    "RISK_CRITICAL",
    "combine",
    "FusionOutput",
]

# Risk levels for HUD and downstream logic
RISK_OK = "OK"
RISK_MONITOR = "MONITOR"
RISK_WARNING = "WARNING"
RISK_CRITICAL = "CRITICAL"


class FusionOutput(TypedDict):
    risk: str
    reason: str


def combine(
    drowsy_metrics: dict,
    drowsy_alert: dict,
    distraction_state: dict,
) -> FusionOutput:
    """
    Combine drowsiness and distraction into a single risk state.

    Priority (highest wins):
    - CRITICAL: Drowsiness alert active (eyes closed too long).
    - WARNING: Distraction window triggered (sustained distraction).
    - MONITOR: Eyes currently closed but not yet alert, or elevated distraction.
    - OK: Otherwise.

    Args:
        drowsy_metrics: From BlinkDetector.metrics() — ear, blink_count,
            is_closed, closure_duration, closed_eye_secs.
        drowsy_alert: From handle_drowsiness() — alert_active, reason.
        distraction_state: From DistractionState.update() — ratio, triggered,
            mean_prob.

    Returns:
        Dict with risk in (OK, MONITOR, WARNING, CRITICAL) and a short reason.
    """
    alert_active = drowsy_alert.get("alert_active", False)
    is_closed = drowsy_metrics.get("is_closed", False)
    closure_duration = drowsy_metrics.get("closure_duration", 0.0)
    triggered = distraction_state.get("triggered", False)
    mean_prob = distraction_state.get("mean_prob", 0.0)

    # Highest priority: drowsiness alert (eyes closed long enough)
    if alert_active:
        reason = drowsy_alert.get("reason") or "Eyes closed too long"
        return {"risk": RISK_CRITICAL, "reason": reason}

    # Distraction window triggered
    if triggered:
        return {
            "risk": RISK_WARNING,
            "reason": "Distracted (sustained)",
        }

    # Monitor: eyes closed but not yet at alert threshold, or elevated distraction
    if is_closed and closure_duration > 0:
        return {
            "risk": RISK_MONITOR,
            "reason": f"Eyes closed {closure_duration:.1f}s",
        }
    if mean_prob >= 0.5:
        return {
            "risk": RISK_MONITOR,
            "reason": f"Distraction likely ({mean_prob:.0%})",
        }

    return {"risk": RISK_OK, "reason": ""}
