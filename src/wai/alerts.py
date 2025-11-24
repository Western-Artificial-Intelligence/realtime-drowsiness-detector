import time

_last_alert_time: float | None = None


def handle_drowsiness(
    metrics: dict,
    min_closed_sec: float = 1.5,
    cooldown_s: float = 3.0,
) -> dict:
    """Check if driver needs to be alerted for prolonged eye closure.
    
    Args:
        metrics: Blink detector metrics (needs 'is_closed' and 'closure_duration')
        min_closed_sec: How long eyes must be closed to trigger alert
        cooldown_s: Time between alerts to avoid spam
    
    Returns:
        Dictionary with 'alert_active' (bool) and 'reason' (str or None)
    """
    global _last_alert_time
    
    # pull out what we need from metrics
    closure = metrics.get("closure_duration", 0.0)
    is_closed = metrics.get("is_closed", False)
    
    now = time.monotonic()
    alert_active = False
    reason = None
    
    # check if eyes have been closed long enough
    if is_closed and closure >= min_closed_sec:
        # make sure we're not in cooldown period
        if _last_alert_time is None or (now - _last_alert_time) >= cooldown_s:
            alert_active = True
            reason = f"Eyes closed {closure:.1f}s"
            _last_alert_time = now
            
            # TODO: uncomment when we add sound alerts
            # try:
            #     playsound("audio.mp3", block=False)
            # except:
            #     pass
    
    return { 
        "alert_active": alert_active,
        "reason": reason,
    }


def reset_alert_state() -> None:
    """Reset cooldown state - useful for testing or starting new session."""
    global _last_alert_time
    _last_alert_time = None
 
 
