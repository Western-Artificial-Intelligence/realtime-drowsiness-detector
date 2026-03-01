'''Main loop.

Calls all modules in order: camera → landmarks → Signal.py metrics
→ distraction inference → fusion → UI.

Displays frame & handles quitting.
FPS calculation.
'''

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from camera import Camera
from landmarks import FaceMeshDetector, LandmarkFrame as FaceLmFrame
from Signal import BlinkDetector, LandmarkFrame as SigLmFrame
from alerts import handle_drowsiness
import ui

# Optional distraction + fusion (graceful fallback if not available)
try:
    from distraction_model import DistractionClassifier
    from distraction_state import DistractionState
    from fusion import combine
    _DISTRACTION_AVAILABLE = True
except ImportError:
    _DISTRACTION_AVAILABLE = False


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Real-time drowsiness + distraction detection"
    )
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--max-faces", type=int, default=1)
    p.add_argument("--flip", action="store_true", default=True)
    # Dev 1: CLI flags for distraction + fusion
    p.add_argument(
        "--enable-distraction",
        action="store_true",
        help="Enable distraction pipeline (YOLO inference)",
    )
    p.add_argument(
        "--enable-fusion",
        action="store_true",
        help="Enable fusion of drowsiness + distraction into single risk state",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default="models/distraction_yolov8/best.pt",
        help="Path to distraction model (.pt). App runs with dummy if load fails.",
    )
    p.add_argument(
        "--yolo-stride",
        type=int,
        default=3,
        help="Run YOLO inference every N frames (1=every frame, 3=every 3rd).",
    )
    p.add_argument(
        "--debug-distraction",
        action="store_true",
        help="Print raw distraction probs every 30 frames (for debugging class index).",
    )
    return p.parse_args(argv)


def _resolve_model_path(model_path: str) -> Path:
    """Resolve model path relative to project root if needed."""
    p = Path(model_path)
    if p.is_absolute() or p.exists():
        return p
    # Try relative to project root (parent of src/wai)
    root = Path(__file__).resolve().parent.parent.parent
    candidate = root / model_path
    return candidate if candidate.exists() else p


def _create_distraction_classifier(model_path: str, device: str = "cpu"):
    """Create DistractionClassifier or dummy on failure (plug-and-play model)."""
    if not _DISTRACTION_AVAILABLE:
        return None, "MODEL OFF (distraction_model not available)"
    path = _resolve_model_path(model_path)
    if not path.exists():
        return None, "MODEL OFF (file not found)"
    try:
        clf = DistractionClassifier(model_path=str(path), device=device, imgsz=224)
        return clf, None
    except Exception as e:
        return None, f"MODEL OFF ({e!s})"


def _dummy_infer(_frame):
    """Dummy inference when model unavailable. Returns prob_distracted=0."""
    return {"prob_distracted": 0.0, "label": "dummy"}


def _face_crop_for_distraction(frame: np.ndarray, lm_frame: FaceLmFrame | None, padding: float = 0.35) -> np.ndarray | None:
    """
    Crop frame to face region for distraction model. Training data is face-centric;
    full webcam frames cause the model to fire constantly (input mismatch).
    Returns None if no face; caller should reuse last probability.
    """
    if lm_frame is None:
        return None
    landmarks = np.array(lm_frame.landmarks, dtype=np.float32)
    if len(landmarks) < 2:
        return None
    h, w = frame.shape[:2]
    xs = landmarks[:, 0] * w
    ys = landmarks[:, 1] * h
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    bw = x_max - x_min
    bh = y_max - y_min
    pad_w = bw * padding
    pad_h = bh * padding
    x0 = max(0, int(x_min - pad_w))
    y0 = max(0, int(y_min - pad_h))
    x1 = min(w, int(x_max + pad_w))
    y1 = min(h, int(y_max + pad_h))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1].copy()


def to_signal_landmarks(face_frame: FaceLmFrame, timestamp: float) -> SigLmFrame:
    """Convert MediaPipe landmarks to Signal landmarks format."""
    landmarks_array = np.array(face_frame.landmarks, dtype=np.float32)
    landmarks_px = landmarks_array.copy()
    landmarks_px[:, 0] *= face_frame.frame_width
    landmarks_px[:, 1] *= face_frame.frame_height
    return SigLmFrame(landmarks=landmarks_px, timestamp=timestamp)


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

    # Distraction pipeline (optional)
    distraction_clf = None
    distraction_model_off_reason = None
    distraction_state = None
    last_prob_distracted = 0.0
    if args.enable_distraction:
        distraction_clf, distraction_model_off_reason = _create_distraction_classifier(
            args.model_path
        )
        distraction_state = DistractionState(window_size=10, trigger_ratio=0.8)
        if distraction_model_off_reason:
            print(f"[Distraction] {distraction_model_off_reason} — using dummy prob=0")

    session_start = time.time()
    frame_count = 0
    yolo_infer_count = 0  # only increments when we actually run YOLO
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

            # 1. Drowsiness pipeline: landmarks → Signal metrics
            lm_frame = face_mesh.process(frame)
            if lm_frame is not None:
                sig_lm = to_signal_landmarks(lm_frame, timestamp=now)
                blink.update(sig_lm)
            metrics = blink.metrics()

            # 2. Drowsiness alerts
            alert_state = handle_drowsiness(
                metrics,
                min_closed_sec=1.5,
                cooldown_s=3.0,
            )

            # 3. Distraction pipeline: YOLO every N frames on FACE CROP (model trained on face-centric images)
            distraction_out = None
            if args.enable_distraction and distraction_state is not None:
                run_yolo = (frame_count % args.yolo_stride) == 0
                if distraction_clf is not None and run_yolo:
                    crop = _face_crop_for_distraction(frame, lm_frame)
                    if crop is not None:
                        try:
                            do_debug = getattr(args, "debug_distraction", False)
                            out = distraction_clf.infer(crop, return_all_probs=do_debug)
                            last_prob_distracted = out.get("prob_distracted", 0.0)
                            yolo_infer_count += 1
                            if do_debug and yolo_infer_count % 10 == 0 and out.get("all_probs") is not None:
                                print(f"[DEBUG distraction] infer#={yolo_infer_count} prob_distracted={last_prob_distracted:.3f} all_probs={out['all_probs']} label={out.get('label')}")
                        except Exception:
                            last_prob_distracted = 0.0
                    # else: no face detected, keep last_prob_distracted (don't run on full frame)
                elif distraction_clf is None:
                    last_prob_distracted = 0.0  # dummy
                distraction_out = distraction_state.update(last_prob_distracted)
                distraction_out["prob_distracted"] = last_prob_distracted
                distraction_out["label"] = (
                    "MODEL OFF" if distraction_model_off_reason else "ok"
                )
                if distraction_model_off_reason:
                    distraction_out["model_off_reason"] = distraction_model_off_reason

            # 4. Fusion: combine drowsiness + distraction into top risk
            fusion_out = None
            if args.enable_fusion and distraction_out is not None:
                fusion_out = combine(metrics, alert_state, distraction_out)
            elif args.enable_fusion and distraction_out is None:
                # Fusion enabled but distraction disabled — use drowsiness only
                fusion_out = combine(
                    metrics,
                    alert_state,
                    {"triggered": False, "mean_prob": 0.0, "ratio": 0.0},
                )

            # 5. FPS
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # 6. Display
            display_frame = (
                lm_frame.annotated_frame if lm_frame is not None else frame
            )
            display_frame = ui.draw_hud(
                display_frame,
                metrics,
                fps,
                alert_state,
                distraction_state=distraction_out,
                fusion_state=fusion_out,
            )
            cv2.imshow("Drowsiness Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        cv2.destroyAllWindows()

        final_metrics = blink.metrics()
        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)
        print(f"Total blinks: {final_metrics['blink_count']}")
        print(f"Total closed time: {final_metrics['closed_eye_secs']:.2f}s")
        print(f"Session duration: {time.time() - session_start:.2f}s")
        print("=" * 50)
        print(final_metrics)


if __name__ == "__main__":
    main()
