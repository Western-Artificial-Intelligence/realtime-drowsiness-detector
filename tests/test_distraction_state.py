"""
Unit tests for distraction_state.DistractionState window trigger behavior.
"""

import pytest
from src.wai.distraction_state import DistractionState


class TestDistractionStateWindowTrigger:
    """Sliding window and trigger_ratio behavior."""

    def test_trigger_ratio_80_percent_ten_frames(self):
        """With window_size=10, trigger_ratio=0.8: 8+ distracted frames → triggered."""
        state = DistractionState(window_size=10, trigger_ratio=0.8, prob_threshold=0.5)
        # Fill window with 3 not-distracted + 7 distracted → ratio 0.7 → not triggered
        for _ in range(3):
            state.update(0.4)
        for _ in range(7):
            out = state.update(0.6)
        assert out["triggered"] is False
        assert out["ratio"] == 0.7
        # one more distracted → 8/10 = 0.8 → triggered
        out = state.update(0.6)
        assert out["triggered"] is True
        assert out["ratio"] == 0.8
        out = state.update(0.6)
        assert out["triggered"] is True

    def test_trigger_ratio_below_threshold_no_trigger(self):
        """Fewer than trigger_ratio of frames above prob_threshold → not triggered."""
        state = DistractionState(window_size=10, trigger_ratio=0.8, prob_threshold=0.5)
        # 6 high, 4 low
        for _ in range(6):
            state.update(0.9)
        for _ in range(4):
            out = state.update(0.1)
        assert out["triggered"] is False
        assert out["ratio"] == 0.6
        assert 0.4 < out["mean_prob"] < 0.6

    def test_mean_prob_correct(self):
        """mean_prob is average of last window_size values."""
        state = DistractionState(window_size=4, trigger_ratio=1.0, prob_threshold=0.5)
        state.update(0.2)
        state.update(0.4)
        state.update(0.6)
        out = state.update(0.8)
        assert out["mean_prob"] == pytest.approx(0.5)

    def test_window_slides_old_frames_dropped(self):
        """After window is full, oldest frame is dropped; trigger can turn off."""
        state = DistractionState(window_size=3, trigger_ratio=2 / 3, prob_threshold=0.5)
        state.update(0.9)  # 1/1
        state.update(0.9)  # 2/2
        out = state.update(0.9)  # 3/3 → triggered
        assert out["triggered"] is True
        # Push one low: window becomes [0.9, 0.9, 0.1] → 2/3 still triggered
        out = state.update(0.1)
        assert out["triggered"] is True
        assert out["ratio"] == pytest.approx(2 / 3)
        # One more low: [0.9, 0.1, 0.1] → 1/3 → not triggered
        out = state.update(0.1)
        assert out["triggered"] is False
        assert out["ratio"] == pytest.approx(1 / 3)

    def test_first_frames_ratio_based_on_filled_slots(self):
        """Before window is full, ratio uses count of frames so far."""
        state = DistractionState(window_size=10, trigger_ratio=0.8, prob_threshold=0.5)
        out = state.update(0.9)  # 1 frame, 1 distracted
        assert out["ratio"] == 1.0
        assert out["mean_prob"] == 0.9
        out = state.update(0.1)
        assert out["ratio"] == 0.5
        assert out["mean_prob"] == 0.5

    def test_reset_clears_window(self):
        """reset() clears history; next update starts fresh."""
        state = DistractionState(window_size=3, trigger_ratio=1.0, prob_threshold=0.5)
        state.update(0.9)
        state.update(0.9)
        state.update(0.9)
        out = state.update(0.9)
        assert out["triggered"] is True
        state.reset()
        out = state.update(0.1)
        assert out["ratio"] == 0.0
        assert out["triggered"] is False
        assert out["mean_prob"] == 0.1

    def test_prob_clamped_to_valid_range(self):
        """Out-of-range probabilities are clamped to [0, 1]."""
        state = DistractionState(window_size=1, trigger_ratio=0.5, prob_threshold=0.5)
        out = state.update(1.5)
        assert out["mean_prob"] == 1.0
        out = state.update(-0.1)
        assert out["mean_prob"] == 0.0

    def test_invalid_init_raises(self):
        """Invalid window_size, trigger_ratio, or prob_threshold raise ValueError."""
        with pytest.raises(ValueError, match="window_size"):
            DistractionState(window_size=0)
        with pytest.raises(ValueError, match="trigger_ratio"):
            DistractionState(trigger_ratio=1.5)
        with pytest.raises(ValueError, match="prob_threshold"):
            DistractionState(prob_threshold=-0.1)
