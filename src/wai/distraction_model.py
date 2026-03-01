"""
Load and run YOLOv8 classification model for distraction detection.

Uses ultralytics YOLO (required for .pt checkpoints from YOLOv8 training).
"""

import time
from typing import Any, Dict, List, Union

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


class DistractionClassifier:
    """
    Load and run inference with a YOLOv8 classification distraction model.

    Contract: infer(frame_bgr) -> {"prob_distracted": float, "label": str}
    """

    def __init__(self, model_path: str, device: str = "cpu", imgsz: int = 224):
        """
        Args:
            model_path: Path to the trained .pt model (YOLOv8 classification).
            device: "cpu" or "cuda".
            imgsz: Input size (e.g. 224).
        """
        if not _YOLO_AVAILABLE:
            raise RuntimeError("ultralytics is required: pip install ultralytics")
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        # Class names from the model (e.g. {0: "not_distracted", 1: "distracted"})
        self.names = getattr(self.model, "names", None) or {}
        # Resolve which index is "distracted" by name (avoids wrong index if folder order differs)
        self._distracted_idx = self._resolve_distracted_index()

    def _resolve_distracted_index(self) -> int | None:
        """Return class index for 'distracted', or None if not found."""
        return self.resolve_distracted_index_from_names(self.names)

    @staticmethod
    def resolve_distracted_index_from_names(names: Union[Dict[int, str], List[str]]) -> int | None:
        """Return class index for 'distracted' from a names dict or list. Used for tests."""
        items = names.items() if isinstance(names, dict) else enumerate(names)
        for idx, name in items:
            if isinstance(name, str) and name.strip().lower() == "distracted":
                return int(idx)
        if len(names) == 2:
            return 1
        return None

    def infer(self, frame_bgr: np.ndarray, return_all_probs: bool = False) -> Dict[str, Any]:
        """
        Run inference on a BGR frame.

        Returns:
            {"prob_distracted": float, "label": str} and optionally "all_probs" for debug.
        """
        if frame_bgr is None:
            out = {"prob_distracted": 0.0, "label": "invalid"}
            if return_all_probs:
                out["all_probs"] = []
            return out

        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results or len(results) == 0:
            out = {"prob_distracted": 0.0, "label": "unknown"}
            if return_all_probs:
                out["all_probs"] = []
            return out

        probs = results[0].probs
        if probs is None:
            out = {"prob_distracted": 0.0, "label": "unknown"}
            if return_all_probs:
                out["all_probs"] = []
            return out

        data = probs.data.cpu().numpy()
        top_idx = int(probs.top1)
        label = self.names.get(top_idx, str(top_idx))
        if self._distracted_idx is not None and self._distracted_idx < len(data):
            prob_distracted = float(data[self._distracted_idx])
        elif len(data) == 2:
            prob_distracted = float(data[1])
        else:
            prob_distracted = float(data[top_idx])

        out = {"prob_distracted": prob_distracted, "label": label}
        if return_all_probs:
            out["all_probs"] = [float(x) for x in data]
        return out

    def benchmark_inference(self, sample_images: List[np.ndarray], iters: int = 50) -> List[float]:
        """Return list of inference times in ms per frame."""
        times_ms = []
        for frame_bgr in sample_images:
            if frame_bgr is None:
                continue
            t0 = time.perf_counter()
            self.infer(frame_bgr)
            times_ms.append((time.perf_counter() - t0) * 1000)
        return times_ms


if __name__ == "__main__":
    if not _YOLO_AVAILABLE:
        print("Install ultralytics: pip install ultralytics")
    else:
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent.parent
        model_path = root / "models" / "distraction_yolov8" / "best.pt"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
        else:
            clf = DistractionClassifier(str(model_path))
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            out = clf.infer(dummy)
            print("Inference OK:", out)
