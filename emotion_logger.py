"""
EmotionLogger: Time-based emotion logging and aggregation system.

Logs detected emotions every second and determines the most prevalent mood
after 60 seconds without disrupting real-time video processing.
"""

import time
from typing import Dict, List, Optional


class EmotionLogger:
    """Logs emotions at regular intervals and aggregates over time periods."""

    # Emotion names mapping (matches emotion.py)
    EMOTIONS = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }

    def __init__(self, log_interval: float = 1.0, aggregation_period: float = 60.0):
        """
        Initialize the EmotionLogger.

        Args:
            log_interval: Time in seconds between emotion logs (default: 1.0)
            aggregation_period: Time in seconds for aggregation period (default: 60.0)
        """
        self.log_interval = log_interval
        self.aggregation_period = aggregation_period
        self.last_log_time = time.time()
        self.cycle_start_time = time.time()

        # Store emotions for aggregation period
        self.emotion_history: List[Dict] = []

        # Store last detected emotion (updated every frame)
        self.current_emotion: Optional[int] = None
        self.current_confidence: float = 0.0

    def update_current_emotion(self, emotion_id: int, confidence: float) -> None:
        """
        Update the current emotion state (called every frame).

        Args:
            emotion_id: Detected emotion ID (0-6)
            confidence: Confidence score (0.0-1.0)
        """
        self.current_emotion = emotion_id
        self.current_confidence = confidence

    def should_log(self) -> bool:
        """
        Check if enough time has passed to log an emotion.

        Returns:
            True if should log, False otherwise
        """
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self.last_log_time = current_time
            return True
        return False

    def log_emotion(self) -> None:
        """Log the current emotion with timestamp to history."""
        if self.current_emotion is not None:
            timestamp = time.time()
            emotion_name = self.EMOTIONS.get(self.current_emotion, 'Unknown')

            self.emotion_history.append({
                'emotion_id': self.current_emotion,
                'emotion_name': emotion_name,
                'confidence': self.current_confidence,
                'timestamp': timestamp
            })

            print(f"[{timestamp:.2f}] Logged: {emotion_name} ({self.current_confidence*100:.1f}%)")

    def should_aggregate(self) -> bool:
        """
        Check if enough time has passed to aggregate emotions.

        Returns:
            True if aggregation period is complete, False otherwise
        """
        return time.time() - self.cycle_start_time >= self.aggregation_period

    def get_prevalent_mood(self) -> Optional[Dict]:
        """
        Determine the most prevalent emotion from the history.

        Returns:
            Dictionary with prevalent mood information, or None if insufficient data
        """
        # Require minimum 10 samples (10 seconds of detection)
        if len(self.emotion_history) < 10:
            print(f"WARNING: Insufficient data ({len(self.emotion_history)} samples, need at least 10)")
            return None

        if not self.emotion_history:
            print("WARNING: No emotions logged in aggregation period")
            return None

        # Count emotion occurrences
        emotion_counts = {}
        for entry in self.emotion_history:
            eid = entry['emotion_id']
            emotion_counts[eid] = emotion_counts.get(eid, 0) + 1

        # Find emotions with maximum count (handle ties)
        max_count = max(emotion_counts.values())
        tied_emotions = [eid for eid, count in emotion_counts.items() if count == max_count]

        # Break ties using average confidence
        if len(tied_emotions) > 1:
            avg_confidences = {}
            for eid in tied_emotions:
                entries = [e for e in self.emotion_history if e['emotion_id'] == eid]
                avg_confidences[eid] = sum(e['confidence'] for e in entries) / len(entries)

            prevalent_emotion_id = max(avg_confidences, key=avg_confidences.get)
        else:
            prevalent_emotion_id = tied_emotions[0]

        # Calculate average confidence for prevalent emotion
        prevalent_entries = [e for e in self.emotion_history if e['emotion_id'] == prevalent_emotion_id]
        avg_confidence = sum(e['confidence'] for e in prevalent_entries) / len(prevalent_entries)

        return {
            'emotion_id': prevalent_emotion_id,
            'emotion_name': self.EMOTIONS[prevalent_emotion_id],
            'count': emotion_counts[prevalent_emotion_id],
            'avg_confidence': avg_confidence,
            'total_samples': len(self.emotion_history)
        }

    def reset_cycle(self) -> None:
        """Clear history and start new aggregation cycle."""
        self.emotion_history = []
        self.cycle_start_time = time.time()
        print(f"\nStarting new 60-second monitoring cycle...\n")
