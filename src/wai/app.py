'''Main loop.

Calls all modules in order.

Displays frame & handles quitting.

FPS calculation.'''

import argparse
from camera import Camera
from landmarks import FaceMeshDetector, LandmarkFrame as FaceLmFrame
from Signal import BlinkDetector, LandmarkFrame as SigLmFrame
import cv2
import time
import numpy as np
from alerts import handle_drowsiness
import ui
import sys
import site

sys.path.append(site.getusersitepackages())

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Real-time drowsiness detection")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--max-faces", type=int, default=1)
    p.add_argument("--flip", action="store_true", default=True)
    return p.parse_args(argv)

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
    args = parse_args()
    cam = Camera(index=args.camera)
    face_mesh = FaceMeshDetector(max_faces=args.max_faces)
    blink = BlinkDetector(
        enter_th=0.23,
        exit_th=0.26,
        min_frames=3,
        ema_alpha=0.3,
    )

    session_start = time.time()
    frame_count = 0
    fps_start_time = time.time()
    fps = 0

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue

            if args.flip:
                frame = cv2.flip(frame, 1)

            now = time.time() - session_start

            lm_frame = face_mesh.process(frame)
            
            # Update blink detector only if landmarks detected
            if lm_frame is not None:
                sig_lm = to_signal_landmarks(lm_frame, timestamp=now)
                blink.update(sig_lm)
            
            # Get metrics (always available, even if no face detected)
            metrics = blink.metrics()
            
            # Handle drowsiness alerts
            alert_state = handle_drowsiness(
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
            
            
            # 1. Choose base frame (annotated or raw)
            display_frame = lm_frame.annotated_frame if lm_frame is not None else frame
            
            # 2. Draw the HUD (Uncomment and format correctly)
            display_frame = ui.draw_hud(
                display_frame,
                metrics,
                fps,
                alert_state
            )
            
            # 3. Show the final combined frame
            cv2.imshow('Drowsiness Detection', display_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:  # ctrl + c
        pass
    finally:
        cam.release()
        cv2.destroyAllWindows()

        image.pngfinal_metrics = blink.metrics()
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Total blinks: {final_metrics['blink_count']}")
        print(f"Total closed time: {final_metrics['closed_eye_secs']:.2f}s")
        print(f"Session duration: {time.time() - session_start:.2f}s")
        print("="*50)
        print(final_metrics)

if __name__ == "__main__":
    main()