"""
Temporal smoothing for distraction detection.

Sliding window over per-frame distraction probabilities so alerts
trigger only when a high fraction of recent frames are distracted
(reduces flicker and false positives).
"""

from collections import deque
from typing import TypedDict

__all__ = ["DistractionState", "DistractionStateOutput"]


class DistractionStateOutput(TypedDict):
    ratio: float
    triggered: bool
    mean_prob: float


class DistractionState:
    """
    Maintains a sliding window of distraction probabilities and
    reports ratio (fraction of frames above threshold), mean probability,
    and whether the trigger threshold is exceeded.
    """

    def __init__(
        self,
        window_size: int = 10,
        trigger_ratio: float = 0.8,
        prob_threshold: float = 0.5,
    ):
        """
        Args:
            window_size: Number of recent frames to consider.
            trigger_ratio: Fraction of window that must be "distracted"
                (prob >= prob_threshold) to set triggered=True (e.g. 0.8 = 80%).
            prob_threshold: Per-frame probability above which a frame
                counts as "distracted" for ratio calculation.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not (0.0 <= trigger_ratio <= 1.0):
            raise ValueError("trigger_ratio must be in [0, 1]")
        if not (0.0 <= prob_threshold <= 1.0):
            raise ValueError("prob_threshold must be in [0, 1]")

        self._window_size = window_size
        self._trigger_ratio = trigger_ratio
        self._prob_threshold = prob_threshold
        self._probs: deque[float] = deque(maxlen=window_size)

    def update(self, prob_distracted: float) -> DistractionStateOutput:
        """
        Push a new per-frame distraction probability and return
        window stats.

        Args:
            prob_distracted: Probability from classifier for this frame [0, 1].
                Clamped to [0, 1] if out of range.

        Returns:
            Dict with:
            - ratio: Fraction of frames in window with prob >= prob_threshold.
            - triggered: True if ratio >= trigger_ratio.
            - mean_prob: Mean of probabilities in the window.
        """
        p = max(0.0, min(1.0, float(prob_distracted)))
        self._probs.append(p)

        n = len(self._probs)
        if n == 0:
            return {
                "ratio": 0.0,
                "triggered": False,
                "mean_prob": 0.0,
            }

        distracted_count = sum(1 for x in self._probs if x >= self._prob_threshold)
        ratio = distracted_count / n
        mean_prob = sum(self._probs) / n
        triggered = ratio >= self._trigger_ratio

        return {
            "ratio": ratio,
            "triggered": triggered,
            "mean_prob": mean_prob,
        }

    def reset(self) -> None:
        """Clear the window (e.g. new session)."""
        self._probs.clear()