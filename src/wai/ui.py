"""HUD overlay for drowsiness + distraction detection."""

import cv2
import numpy as np


def draw_hud(
    frame,
    metrics,
    fps,
    alert_state,
    distraction_state=None,
    fusion_state=None,
):
    """Draw HUD with drowsiness, distraction, and risk state.

    Args:
        frame: BGR image to draw on.
        metrics: From BlinkDetector.metrics() — ear, blink_count, is_closed,
            closure_duration, closed_eye_secs.
        fps: Current FPS.
        alert_state: From handle_drowsiness() — alert_active, level, color.
        distraction_state: Optional. From DistractionState.update() plus
            prob_distracted, label, model_off_reason.
        fusion_state: Optional. From fusion.combine() — risk, reason.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]

    # Use fusion risk when available, else alert level
    if fusion_state is not None:
        risk = fusion_state.get("risk", "OK")
        reason = fusion_state.get("reason", "")
        level = risk
        color = _risk_color(risk)
    else:
        level = alert_state.get("level", "NORMAL")
        color = alert_state.get("color", (0, 255, 0))

    # Glass header
    header_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    font = cv2.FONT_HERSHEY_DUPLEX

    # Left: Drowsiness metrics
    ear_val = metrics.get("ear", 0.0)
    blinks = metrics.get("blink_count", 0)
    closure = metrics.get("closure_duration", 0.0)
    cv2.putText(frame, f"EAR: {ear_val:.3f}", (25, 40), font, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"BLINKS: {blinks}", (25, 60), font, 0.7, (255, 255, 255), 1)
    cv2.putText(
        frame, f"CLOSURE: {closure:.1f}s", (25, 80), font, 0.6, (255, 255, 255), 1
    )

    # Right: Distraction (if enabled) + FPS
    y_right = 40
    if distraction_state is not None:
        prob = distraction_state.get("prob_distracted", 0.0)
        ratio = distraction_state.get("ratio", 0.0)
        triggered = distraction_state.get("triggered", False)
        model_off = distraction_state.get("model_off_reason")
        if model_off:
            cv2.putText(
                frame, "DISTR: MODEL OFF", (w - 200, y_right), font, 0.55, (128, 128, 128), 1
            )
        else:
            cv2.putText(
                frame,
                f"DISTR: {prob:.0%}",
                (w - 200, y_right),
                font,
                0.6,
                (255, 200, 100),
                1,
            )
            y_right += 20
            cv2.putText(
                frame,
                f"WIN: {ratio:.0%} {'!' if triggered else ''}",
                (w - 200, y_right),
                font,
                0.55,
                (255, 200, 100),
                1,
            )
            y_right += 20
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, y_right + 20), font, 0.7, (0, 255, 255), 1)

    # Center: Risk / mode
    status_text = f"MODE: {level}"
    text_size = cv2.getTextSize(status_text, font, 0.9, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, status_text, (text_x, 55), font, 0.9, color, 2)

    # Alert border + banner
    if level != "OK" and level != "NORMAL":
        thickness = 10 if level == "CRITICAL" else 4
        cv2.rectangle(frame, (0, 0), (w, h), color, thickness)

        if level == "CRITICAL":
            banner_w, banner_h = 400, 100
            bx, by = (w - banner_w) // 2, (h - banner_h) // 2
            sub_img = frame[by : by + banner_h, bx : bx + banner_w]
            white_rect = np.full(sub_img.shape, color, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
            frame[by : by + banner_h, bx : bx + banner_w] = res
            cv2.putText(
                frame, "PULL OVER", (bx + 65, by + 65), font, 1.5, (255, 255, 255), 3
            )

    return frame


def _risk_color(risk: str):
    """Map risk level to BGR color."""
    if risk == "CRITICAL":
        return (0, 0, 255)
    if risk == "WARNING":
        return (0, 255, 255)
    if risk == "MONITOR":
        return (0, 255, 200)
    return (0, 255, 0)
