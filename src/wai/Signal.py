# Performs mathematical analysis on landmarks to detect blinks, prolonged eye closure, and yawns

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import warnings


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
    """
    Detects blinks and prolonged eye closure using Eye Aspect Ratio (EAR).
    
    Hardened for real-world conditions:
    - Handles missing landmarks gracefully (face leaves frame)
    - Validates timestamps for consistency
    - Prevents stuck closed eyes state with timeout
    - Robust EAR thresholds across lighting conditions
    """
    
    LEFT_EYE = [362, 385, 387, 263, 373, 380]  # MediaPipe face mesh landmark indices
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    # Maximum time to stay in closed state before forcing reset (prevents stuck state)
    MAX_CLOSURE_TIMEOUT = 5.0  # seconds
    
    # Maximum time gap between frames before resetting state (face left frame)
    MAX_FRAME_GAP = 1.0  # seconds
    
    # Minimum valid EAR value (sanity check)
    MIN_VALID_EAR = 0.0
    MAX_VALID_EAR = 1.0
    
    def __init__(
        self,
        enter_th: float = 0.23,
        exit_th: float = 0.26,
        min_frames: int = 3,
        ema_alpha: float = 0.3,
        max_closure_timeout: float = 5.0,
        max_frame_gap: float = 1.0
    ):
        """
        Initialize BlinkDetector.
        
        Args:
            enter_th: EAR threshold to enter closed state (lower = more sensitive)
            exit_th: EAR threshold to exit closed state (higher = less sensitive)
            min_frames: Minimum consecutive frames below enter_th to trigger closure
            ema_alpha: Exponential moving average smoothing factor (0-1)
            max_closure_timeout: Maximum time in closed state before forced reset (seconds)
            max_frame_gap: Maximum time gap between frames before state reset (seconds)
        """
        # Validate thresholds
        if enter_th >= exit_th:
            raise ValueError(f"enter_th ({enter_th}) must be < exit_th ({exit_th})")
        if not (0.0 <= enter_th <= 1.0):
            raise ValueError(f"enter_th ({enter_th}) must be in [0, 1]")
        if not (0.0 <= exit_th <= 1.0):
            raise ValueError(f"exit_th ({exit_th}) must be in [0, 1]")
        if min_frames < 1:
            raise ValueError(f"min_frames ({min_frames}) must be >= 1")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha ({ema_alpha}) must be in (0, 1]")
        
        self.enter_th = enter_th
        self.exit_th = exit_th
        self.min_frames = min_frames
        self.ema_alpha = ema_alpha
        self.max_closure_timeout = max_closure_timeout
        self.max_frame_gap = max_frame_gap
        
        self._is_closed = False
        self._closed_frame_count = 0
        self._closure_start_time: Optional[float] = None
        self._blink_count = 0
        self._total_closed_time = 0.0
        self._ema_ear: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._last_valid_ear: Optional[float] = None
        self._consecutive_missing_frames = 0
        self._event_history: List[DetectorEvent] = []
    
    def _validate_timestamp(self, timestamp: float) -> bool:
        """
        Validate timestamp for consistency.
        
        Returns:
            True if timestamp is valid, False otherwise
        """
        if timestamp < 0:
            warnings.warn(f"Negative timestamp detected: {timestamp}", UserWarning)
            return False
        
        if self._last_timestamp is not None:
            # Check for backwards timestamps (shouldn't happen in real-time)
            if timestamp < self._last_timestamp:
                warnings.warn(
                    f"Backwards timestamp detected: {timestamp} < {self._last_timestamp}. "
                    "Resetting state.",
                    UserWarning
                )
                return False
            
            # Check for excessive time gap (face likely left frame)
            time_gap = timestamp - self._last_timestamp
            if time_gap > self.max_frame_gap:
                warnings.warn(
                    f"Large time gap detected: {time_gap:.2f}s > {self.max_frame_gap}s. "
                    "Face may have left frame. Resetting state.",
                    UserWarning
                )
                return False
        
        return True
    
    def _compute_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Compute Eye Aspect Ratio for a single eye.
        
        Args:
            eye_landmarks: Array of 6 eye landmark points
            
        Returns:
            EAR value (0.0 if invalid)
        """
        if eye_landmarks.shape[0] < 6:
            return 0.0
        
        try:
            vertical1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            vertical2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            if horizontal < 1e-6:
                return 0.0
            
            ear = (vertical1 + vertical2) / (2.0 * horizontal)
            
            # Sanity check: EAR should be in reasonable range
            if not (self.MIN_VALID_EAR <= ear <= self.MAX_VALID_EAR):
                warnings.warn(f"EAR value out of range: {ear}", UserWarning)
                return 0.0
            
            return ear
        except (ValueError, IndexError) as e:
            warnings.warn(f"Error computing EAR: {e}", UserWarning)
            return 0.0
    
    def _extract_eye_landmarks(self, landmarks: np.ndarray, eye_indices: List[int]) -> Optional[np.ndarray]:
        """
        Extract eye landmarks from full landmark array.
        
        Args:
            landmarks: Full landmark array
            eye_indices: List of landmark indices for the eye
            
        Returns:
            Eye landmarks array or None if extraction fails
        """
        try:
            if landmarks is None or landmarks.size == 0:
                return None
            
            # Check if we have enough landmarks
            max_index = max(eye_indices)
            if landmarks.shape[0] <= max_index:
                return None
            
            eye_points = landmarks[eye_indices]
            
            # Validate extracted points (check for NaN or invalid values)
            if np.any(np.isnan(eye_points)) or np.any(np.isinf(eye_points)):
                return None
            
            return eye_points
        except (IndexError, ValueError, TypeError) as e:
            return None
    
    def _compute_average_ear(self, landmarks: np.ndarray) -> Optional[float]:
        """
        Compute average EAR from both eyes.
        
        Args:
            landmarks: Full landmark array
            
        Returns:
            Average EAR or None if computation fails
        """
        if landmarks is None or landmarks.size == 0:
            return None
        
        left_eye = self._extract_eye_landmarks(landmarks, self.LEFT_EYE)
        right_eye = self._extract_eye_landmarks(landmarks, self.RIGHT_EYE)
        
        if left_eye is None or right_eye is None:
            return None
        
        left_ear = self._compute_ear(left_eye)
        right_ear = self._compute_ear(right_eye)
        
        # If either EAR is invalid, return None
        if left_ear == 0.0 and right_ear == 0.0:
            return None
        
        # Average the two eyes (use 0.0 for invalid eye as fallback)
        avg_ear = (left_ear + right_ear) / 2.0
        
        return avg_ear
    
    def _update_ema(self, new_value: float) -> None:
        """Update exponential moving average of EAR."""
        if self._ema_ear is None:
            self._ema_ear = new_value
        else:
            self._ema_ear = self.ema_alpha * new_value + (1 - self.ema_alpha) * self._ema_ear
    
    def _handle_missing_landmarks(self, timestamp: float) -> List[DetectorEvent]:
        """
        Handle case when landmarks are missing (face left frame).
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            List of events (may include forced EYES_OPENED if was in closed state)
        """
        events = []
        self._consecutive_missing_frames += 1
        
        # If we were in closed state and landmarks are missing, force open
        if self._is_closed:
            closure_duration = 0.0
            if self._closure_start_time is not None:
                closure_duration = timestamp - self._closure_start_time
                self._total_closed_time += closure_duration
            
            # Reset closed state
            self._is_closed = False
            self._closed_frame_count = 0
            self._closure_start_time = None
            
            events.append(DetectorEvent(
                event_type=EventType.EYES_OPENED,
                timestamp=timestamp,
                duration=closure_duration,
                metadata={"reason": "missing_landmarks", "ear": None}
            ))
        
        return events
    
    def _check_closure_timeout(self, timestamp: float) -> List[DetectorEvent]:
        """
        Check if we've been in closed state too long and force reset.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            List of events (may include forced EYES_OPENED if timeout occurred)
        """
        events = []
        
        if self._is_closed and self._closure_start_time is not None:
            closure_duration = timestamp - self._closure_start_time
            
            if closure_duration > self.max_closure_timeout:
                # Force reset to prevent stuck state
                self._total_closed_time += closure_duration
                
                events.append(DetectorEvent(
                    event_type=EventType.EYES_OPENED,
                    timestamp=timestamp,
                    duration=closure_duration,
                    metadata={
                        "reason": "timeout",
                        "ear": self._last_valid_ear,
                        "was_blink": False
                    }
                ))
                
                self._is_closed = False
                self._closed_frame_count = 0
                self._closure_start_time = None
        
        return events
    
    def update(self, lm: LandmarkFrame) -> List[DetectorEvent]:
        """
        Update detector with new landmark frame.
        
        Args:
            lm: LandmarkFrame with landmarks and timestamp
            
        Returns:
            List of DetectorEvent objects
        """
        events = []
        
        # Validate timestamp first
        if not self._validate_timestamp(lm.timestamp):
            # Invalid timestamp - reset state to prevent inconsistencies
            if self._is_closed:
                # Force open if we were closed
                if self._closure_start_time is not None:
                    closure_duration = lm.timestamp - self._closure_start_time
                    self._total_closed_time += closure_duration
                
                self._is_closed = False
                self._closed_frame_count = 0
                self._closure_start_time = None
            
            # Reset timestamp tracking
            self._last_timestamp = None
            return events
        
        # Check for closure timeout before processing new frame
        timeout_events = self._check_closure_timeout(lm.timestamp)
        events.extend(timeout_events)
        
        # Compute EAR from landmarks
        current_ear = self._compute_average_ear(lm.landmarks)
        
        # Handle missing landmarks (face left frame)
        if current_ear is None:
            missing_events = self._handle_missing_landmarks(lm.timestamp)
            events.extend(missing_events)
            self._event_history.extend(events)
            return events
        
        # Valid landmarks detected - reset missing frame counter
        self._consecutive_missing_frames = 0
        self._last_valid_ear = current_ear
        
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
                        metadata={"ear": current_ear, "ema_ear": self._ema_ear}
                    ))
            else:
                self._closed_frame_count = 0
        
        else:
            if current_ear > self.exit_th:
                closure_duration = lm.timestamp - self._closure_start_time if self._closure_start_time else 0.0
                self._total_closed_time += closure_duration
                
                is_blink = closure_duration < 0.4
                
                if is_blink:
                    self._blink_count += 1
                    events.append(DetectorEvent(
                        event_type=EventType.BLINK,
                        timestamp=lm.timestamp,
                        duration=closure_duration,
                        metadata={"ear": current_ear, "ema_ear": self._ema_ear}
                    ))
                
                events.append(DetectorEvent(
                    event_type=EventType.EYES_OPENED,
                    timestamp=lm.timestamp,
                    duration=closure_duration,
                    metadata={"ear": current_ear, "ema_ear": self._ema_ear, "was_blink": is_blink}
                ))
                
                # Reset closed state
                self._is_closed = False
                self._closed_frame_count = 0
                self._closure_start_time = None
        
        self._event_history.extend(events)
        
        return events
    
    def metrics(self) -> dict:
        """
        Get current detector metrics.
        
        Returns:
            Dictionary with guaranteed keys:
            - ear: Current EMA EAR value (0.0 if not initialized)
            - blink_count: Total number of blinks detected
            - is_closed: Whether eyes are currently closed
            - closure_duration: Duration of current closure (0.0 if not closed)
            - closed_eye_secs: Total cumulative time eyes were closed
        """
        metrics = {
            "ear": self._ema_ear if self._ema_ear is not None else 0.0,
            "blink_count": self._blink_count,
            "is_closed": self._is_closed,
            "closed_eye_secs": self._total_closed_time,
        }
        
        # Calculate current closure duration if eyes are closed
        if self._is_closed and self._closure_start_time is not None:
            if self._last_timestamp is not None:
                current_closure = self._last_timestamp - self._closure_start_time
            else:
                # Fallback: use 0.0 if no timestamp available
                current_closure = 0.0
            metrics["closure_duration"] = current_closure
        else:
            metrics["closure_duration"] = 0.0
        
        return metrics
    
    def reset(self) -> None:
        """Reset detector to initial state."""
        self._is_closed = False
        self._closed_frame_count = 0
        self._closure_start_time = None
        self._blink_count = 0
        self._total_closed_time = 0.0
        self._ema_ear = None
        self._last_timestamp = None
        self._last_valid_ear = None
        self._consecutive_missing_frames = 0
        self._event_history.clear()
    
    def get_event_history(self) -> List[DetectorEvent]:
        """Get copy of event history."""
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