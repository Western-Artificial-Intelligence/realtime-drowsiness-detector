import cv2
import numpy as np

def draw_hud(frame, metrics, fps, alert_state):
    h, w = frame.shape[:2]
    color = alert_state.get("color", (0, 255, 0))
    level = alert_state.get("level", "NORMAL")
    
    # 1. Create a "Glass" Header Panel (Sleeker Transparency)
    # This creates a dark, 30% transparent bar at the top for high-contrast text
    header_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # 2. Modern Typography (Duplex font is thinner and cleaner)
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Left Side: Key Driver Metrics
    ear_val = metrics.get('ear', 0.0)
    blinks = metrics.get('blink_count', 0)
    cv2.putText(frame, f"EAR: {ear_val:.3f}", (25, 40), font, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"BLINKS: {blinks}", (25, 70), font, 0.7, (255, 255, 255), 1)

    # Right Side: Performance
    cv2.putText(frame, f"System: {int(fps)} FPS", (w - 180, 55), font, 0.7, (0, 255, 255), 1)

    # 3. Centered Status Pill
    # Instead of a giant block of text, we center the status for a balanced look
    status_text = f"MODE: {level}"
    text_size = cv2.getTextSize(status_text, font, 0.9, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, status_text, (text_x, 55), font, 0.9, color, 2)

    # 4. High-Impact Alerts (Only shows when needed)
    if level != "NORMAL":
        # Draw a thin, pulsing-style border
        thickness = 10 if level == "CRITICAL" else 4
        cv2.rectangle(frame, (0, 0), (w, h), color, thickness)
        
        if level == "CRITICAL":
            # "Glass" Alert Banner in the center
            banner_w, banner_h = 400, 100
            bx, by = (w - banner_w)//2, (h - banner_h)//2
            sub_img = frame[by:by+banner_h, bx:bx+banner_w]
            white_rect = np.full(sub_img.shape, color, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
            frame[by:by+banner_h, bx:bx+banner_w] = res
            
            cv2.putText(frame, "PULL OVER", (bx + 65, by + 65), font, 1.5, (255, 255, 255), 3)

    return frame