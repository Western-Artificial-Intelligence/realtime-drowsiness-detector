import time

_last_alert_time: float | None = None

def handle_drowsiness(
    metrics: dict,
    min_closed_sec: float = 1.5,
    cooldown_s: float = 3.0,
) -> dict:
    """Check if driver needs to be alerted for prolonged eye closure with graded severity."""
    global _last_alert_time
    
    closure = metrics.get("closure_duration", 0.0)
    is_closed = metrics.get("is_closed", False)
    
    now = time.monotonic()
    alert_active = False
    reason = None
    
    # --- ADDED: Severity Levels & Colors ---
    level = "NORMAL"
    color = (0, 255, 0)  # Green

    if is_closed:
        # Check for Critical first
        if closure >= min_closed_sec:
            level = "CRITICAL"
            color = (0, 0, 255)  # Red
            
            if _last_alert_time is None or (now - _last_alert_time) >= cooldown_s:
                alert_active = True
                reason = f"Eyes closed {closure:.1f}s"
                _last_alert_time = now
        # Check for Warning
        elif closure >= 0.5:
            level = "WARNING"
            color = (0, 255, 255)  # Yellow
            reason = "Warning: Drowsy"
    
    return { 
        "alert_active": alert_active,
        "reason": reason,
        "level": level,    # UI  use this
        "color": color     # UI  use this
    }

def reset_alert_state() -> None:
    global _last_alert_time
    _last_alert_time = None
