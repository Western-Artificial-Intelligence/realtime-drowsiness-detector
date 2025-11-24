"""
app.py
Main application loop integrating camera, landmarks, signals, alerts, and UI.
"""

import cv2
import time
import numpy as np
from camera import Camera
from landmarks import FaceMeshDetector, LandmarkFrame as FaceLmFrame
from Signal import BlinkDetector, LandmarkFrame as SigLmFrame
import alerts
import ui


def to_signal_landmarks(face_frame: FaceLmFrame, timestamp: float) -> SigLmFrame:
    """
    Convert MediaPipe landmarks to Signal landmarks format.
    
    Args:
        face_frame: FaceLmFrame from FaceMeshDetector
        timestamp: Current timestamp in seconds
        
    Returns:
        SigLmFrame compatible with BlinkDetector
    """
   
    landmarks_array = np.array(face_frame.landmarks, dtype=np.float32)

    landmarks_px = landmarks_array.copy()
    landmarks_px[:, 0] *= face_frame.frame_width
    landmarks_px[:, 1] *= face_frame.frame_height
    
    return SigLmFrame(
        landmarks=landmarks_px,
        timestamp=timestamp
    )


def main():
    """Main application loop."""
    
   
    camera = Camera(camera_id=0)
    face_detector = FaceMeshDetector(
        max_faces=1,
        det_conf=0.5,
        track_conf=0.5
    )
    blink_detector = BlinkDetector(
        enter_th=0.23,
        exit_th=0.26,
        min_frames=3,
        ema_alpha=0.3
    )
    
    # FPS tracking
    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0
    

    session_start = time.time()
    
    print("Starting drowsiness detection system...")
    print("Press 'q' to quit")
    
    try:
        while True:
            # Capture frame
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Get current timestamp
            now = time.time() - session_start
            
            # Detect facial landmarks
            lm_frame = face_detector.process(frame)
            
            # Update blink detector only if landmarks detected
            if lm_frame is not None:
                sig_lm = to_signal_landmarks(lm_frame, timestamp=now)
                blink_detector.update(sig_lm)
            
            # Get metrics (always available, even if no face detected)
            metrics = blink_detector.metrics()
            
            # Handle drowsiness alerts
            alert_state = alerts.handle_drowsiness(
                metrics,
                min_closed_sec=1.5,
                cooldown_s=3.0,
            )
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()
            
            
            display_frame = lm_frame.annotated_frame if lm_frame is not None else frame
            
            # Draw HUD overlay
            display_frame = ui.draw_hud(
                display_frame,
                metrics,
                fps,
                alert_state
            )
            
            # Show frame
            cv2.imshow('Drowsiness Detection', display_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        final_metrics = blink_detector.metrics()
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Total blinks: {final_metrics['blink_count']}")
        print(f"Total closed time: {final_metrics['closed_eye_secs']:.2f}s")
        print(f"Session duration: {time.time() - session_start:.2f}s")
        print("="*50)


if __name__ == "__main__":
    main()
