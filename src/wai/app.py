'''Main loop.

Calls all modules in order.

Displays frame & handles quitting.

FPS calculation.'''

import argparse
from .camera import Camera
from .landmarks import FaceMeshDetector
from .Signal import BlinkDetector
import cv2
import time

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Real-time drowsiness detection")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--max-faces", type=int, default=1)
    p.add_argument("--flip", action="store_true", default=True)
    return p.parse_args(argv)

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

    last_time = time.time()
    
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue

            if args.flip:
                frame = cv2.flip(frame, 1)

            now = time.time()
            lm_frame = face_mesh.process(frame)
            
            # DEV B CODE HERE

            dt = time.time() - last_time
            fps = 1.0 / dt
            last_time = time.time()

            cv2.imshow("Drowsiness Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:  # ctrl + c
        pass
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()