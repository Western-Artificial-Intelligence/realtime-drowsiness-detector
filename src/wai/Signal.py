# Performs mathematical analysis on landmarks to detect blinks, prolonged eye closure, and yawns

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    BLINK = "blink"
    EYES_CLOSED = "eyes_closed"
    EYES_OPENED = "eyes_opened"


@dataclass
class DetectorEvent:
    event_type: EventType
    timestamp: float
    duration: Optional[float] = None  # Duration in seconds for closure events
    metadata: Optional[dict] = None


@dataclass
class LandmarkFrame:
    landmarks: np.ndarray  # Shape: (N, 2) or (N, 3) for 2D/3D landmarks
    timestamp: float
    frame_id: int = 0


class BlinkDetector:
    # Detects blinks and prolonged eye closure using Eye Aspect Ratio (EAR)
    
    LEFT_EYE = [362, 385, 387, 263, 373, 380]  # MediaPipe face mesh landmark indices
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    def __init__(
        self,
        enter_th: float = 0.23,
        exit_th: float = 0.26,
        min_frames: int = 3,
        ema_alpha: float = 0.3
    ):
        self.enter_th = enter_th
        self.exit_th = exit_th
        self.min_frames = min_frames
        self.ema_alpha = ema_alpha
        
        self._is_closed = False
        self._closed_frame_count = 0
        self._closure_start_time: Optional[float] = None
        self._blink_count = 0
        self._total_closed_time = 0.0
        self._ema_ear: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._event_history: List[DetectorEvent] = []
    
    def _compute_ear(self, eye_landmarks: np.ndarray) -> float:
        vertical1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        if horizontal < 1e-6:
            return 0.0
        
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    
    def _extract_eye_landmarks(self, landmarks: np.ndarray, eye_indices: List[int]) -> Optional[np.ndarray]:
        try:
            eye_points = landmarks[eye_indices]
            return eye_points
        except IndexError:
            return None
    
    def _compute_average_ear(self, landmarks: np.ndarray) -> Optional[float]:
        left_eye = self._extract_eye_landmarks(landmarks, self.LEFT_EYE)
        right_eye = self._extract_eye_landmarks(landmarks, self.RIGHT_EYE)
        
        if left_eye is None or right_eye is None:
            return None
        
        left_ear = self._compute_ear(left_eye)
        right_ear = self._compute_ear(right_eye)
        
        return (left_ear + right_ear) / 2.0
    
    def _update_ema(self, new_value: float) -> None:
        if self._ema_ear is None:
            self._ema_ear = new_value
        else:
            self._ema_ear = self.ema_alpha * new_value + (1 - self.ema_alpha) * self._ema_ear
    
    def update(self, lm: LandmarkFrame) -> List[DetectorEvent]:
        events = []
        
        current_ear = self._compute_average_ear(lm.landmarks)
        if current_ear is None:
            return events
        
        self._update_ema(current_ear)
        
        time_delta = 0.0
        if self._last_timestamp is not None:
            time_delta = lm.timestamp - self._last_timestamp
        self._last_timestamp = lm.timestamp
        
        if not self._is_closed:
            if current_ear < self.enter_th:
                self._closed_frame_count += 1
                
                if self._closed_frame_count >= self.min_frames:
                    self._is_closed = True
                    self._closure_start_time = lm.timestamp
                    
                    events.append(DetectorEvent(
                        event_type=EventType.EYES_CLOSED,
                        timestamp=lm.timestamp,
                        metadata={"ear": current_ear}
                    ))
            else:
                self._closed_frame_count = 0
        
        else:
            if current_ear > self.exit_th:
                closure_duration = lm.timestamp - self._closure_start_time if self._closure_start_time else 0.0
                self._total_closed_time += closure_duration
                
                is_blink = closure_duration < 0.4  # Classify as blink if closure was brief
                
                if is_blink:
                    self._blink_count += 1
                    events.append(DetectorEvent(
                        event_type=EventType.BLINK,
                        timestamp=lm.timestamp,
                        duration=closure_duration,
                        metadata={"ear": current_ear}
                    ))
                
                events.append(DetectorEvent(
                    event_type=EventType.EYES_OPENED,
                    timestamp=lm.timestamp,
                    duration=closure_duration,
                    metadata={"ear": current_ear, "was_blink": is_blink}
                ))
                
                self._is_closed = False
                self._closed_frame_count = 0
                self._closure_start_time = None
        
        self._event_history.extend(events)
        
        return events
    
    def metrics(self) -> dict:
        metrics = {
            "blink_count": self._blink_count,
            "closed_eye_secs": self._total_closed_time,
            "ear": self._ema_ear if self._ema_ear is not None else 0.0,
            "is_closed": self._is_closed,
        }
        
        if self._is_closed and self._closure_start_time and self._last_timestamp:
            current_closure = self._last_timestamp - self._closure_start_time
            metrics["closure_duration"] = current_closure
        else:
            metrics["closure_duration"] = 0.0
        
        return metrics
    
    def reset(self) -> None:
        self._is_closed = False
        self._closed_frame_count = 0
        self._closure_start_time = None
        self._blink_count = 0
        self._total_closed_time = 0.0
        self._ema_ear = None
        self._last_timestamp = None
        self._event_history.clear()
    
    def get_event_history(self) -> List[DetectorEvent]:
        return self._event_history.copy()


class YawnDetector:
    # Detects yawns using Mouth Aspect Ratio (MAR) - placeholder implementation
    
    MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409]  # MediaPipe face mesh landmark indices
    MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
    
    def __init__(self, threshold: float = 0.6, min_frames: int = 10):
        self.threshold = threshold
        self.min_frames = min_frames
        self._yawn_count = 0
    
    def update(self, lm: LandmarkFrame) -> List[DetectorEvent]:
        return []
    
    def metrics(self) -> dict:
        return {"yawn_count": self._yawn_count}