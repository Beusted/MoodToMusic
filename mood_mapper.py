"""
Mood Mapper: Maps detected emotions to music emotional dimensions.

Converts facial emotion detection results (7 emotions) to VAD
(Valence-Arousal-Dominance) ranges for music recommendation.

Based on the circumplex model of affect and music emotion research.
"""

from typing import Dict, List, Tuple


# Emotion to VAD mapping with associated music characteristics
EMOTION_TO_VAD_MAPPING = {
    0: {  # Angry
        'emotion': 'Angry',
        'valence': (1.5, 3.5),      # Low valence (negative)
        'arousal': (5.5, 7.5),       # High arousal (intense)
        'dominance': (5.0, 7.0),     # High dominance (powerful)
        'seeds': ['aggressive', 'angry', 'fierce', 'hostile', 'tense', 'defiant', 'intense']
    },
    1: {  # Disgust
        'emotion': 'Disgust',
        'valence': (0.5, 2.5),       # Very low valence
        'arousal': (3.5, 5.5),       # Medium arousal
        'dominance': (4.0, 6.0),     # Medium-high dominance
        'seeds': ['bitter', 'cynical', 'harsh', 'disturbed', 'uneasy', 'dark']
    },
    2: {  # Fear
        'emotion': 'Fear',
        'valence': (1.0, 3.0),       # Low valence
        'arousal': (5.0, 7.0),       # High arousal
        'dominance': (1.0, 3.5),     # Low dominance (submissive)
        'seeds': ['anxious', 'nervous', 'tense', 'scared', 'dark', 'eerie', 'ominous', 'spooky']
    },
    3: {  # Happy
        'emotion': 'Happy',
        'valence': (6.0, 8.5),       # High valence (positive)
        'arousal': (5.0, 7.3),       # Medium-high arousal
        'dominance': (5.5, 7.5),     # High dominance (confident)
        'seeds': ['cheerful', 'happy', 'joyful', 'fun', 'uplifting', 'bright', 'positive',
                  'energetic', 'exciting', 'playful']
    },
    4: {  # Sad
        'emotion': 'Sad',
        'valence': (0.5, 3.0),       # Low valence
        'arousal': (1.0, 3.5),       # Low arousal (low energy)
        'dominance': (2.0, 4.5),     # Low-medium dominance
        'seeds': ['sad', 'melancholy', 'depressing', 'sorrowful', 'gloomy', 'lonely',
                  'bittersweet', 'mellow', 'somber']
    },
    5: {  # Surprise
        'emotion': 'Surprise',
        'valence': (5.0, 7.0),       # Medium-high valence (can be positive)
        'arousal': (6.0, 7.5),       # Very high arousal
        'dominance': (4.0, 6.5),     # Medium dominance
        'seeds': ['surprising', 'exciting', 'dramatic', 'unexpected', 'intense', 'powerful',
                  'epic', 'dynamic']
    },
    6: {  # Neutral
        'emotion': 'Neutral',
        'valence': (4.0, 6.0),       # Medium valence
        'arousal': (3.0, 5.0),       # Medium arousal
        'dominance': (4.0, 6.0),     # Medium dominance
        'seeds': ['calm', 'peaceful', 'relaxed', 'mellow', 'easy', 'soft', 'ambient',
                  'soothing', 'gentle']
    }
}


def get_target_vad(emotion_id: int) -> Dict[str, any]:
    """
    Get target VAD profile for a detected emotion.

    Args:
        emotion_id: Detected emotion ID (0-6)

    Returns:
        Dictionary containing:
            - valence: Target valence value (midpoint of range)
            - arousal: Target arousal value (midpoint of range)
            - dominance: Target dominance value (midpoint of range)
            - valence_range: Tuple of (min, max) valence
            - arousal_range: Tuple of (min, max) arousal
            - dominance_range: Tuple of (min, max) dominance
            - seeds: List of relevant emotion keywords
            - emotion_name: Name of the emotion

    Raises:
        KeyError: If emotion_id is not valid (not 0-6)
    """
    if emotion_id not in EMOTION_TO_VAD_MAPPING:
        raise KeyError(f"Invalid emotion_id: {emotion_id}. Must be 0-6.")

    mapping = EMOTION_TO_VAD_MAPPING[emotion_id]

    return {
        'valence': (mapping['valence'][0] + mapping['valence'][1]) / 2,
        'arousal': (mapping['arousal'][0] + mapping['arousal'][1]) / 2,
        'dominance': (mapping['dominance'][0] + mapping['dominance'][1]) / 2,
        'valence_range': mapping['valence'],
        'arousal_range': mapping['arousal'],
        'dominance_range': mapping['dominance'],
        'seeds': mapping['seeds'],
        'emotion_name': mapping['emotion']
    }


def get_emotion_seeds(emotion_id: int) -> List[str]:
    """
    Get relevant seed keywords for an emotion.

    Args:
        emotion_id: Detected emotion ID (0-6)

    Returns:
        List of seed keywords for the emotion

    Raises:
        KeyError: If emotion_id is not valid (not 0-6)
    """
    if emotion_id not in EMOTION_TO_VAD_MAPPING:
        raise KeyError(f"Invalid emotion_id: {emotion_id}. Must be 0-6.")

    return EMOTION_TO_VAD_MAPPING[emotion_id]['seeds']


def vad_in_range(valence: float, arousal: float, dominance: float,
                  emotion_id: int) -> bool:
    """
    Check if VAD values fall within the range for a given emotion.

    Args:
        valence: Valence value (0-10 scale)
        arousal: Arousal value (0-10 scale)
        dominance: Dominance value (0-10 scale)
        emotion_id: Emotion ID to check against (0-6)

    Returns:
        True if all VAD values are within the emotion's ranges, False otherwise

    Raises:
        KeyError: If emotion_id is not valid (not 0-6)
    """
    if emotion_id not in EMOTION_TO_VAD_MAPPING:
        raise KeyError(f"Invalid emotion_id: {emotion_id}. Must be 0-6.")

    mapping = EMOTION_TO_VAD_MAPPING[emotion_id]

    valence_match = mapping['valence'][0] <= valence <= mapping['valence'][1]
    arousal_match = mapping['arousal'][0] <= arousal <= mapping['arousal'][1]
    dominance_match = mapping['dominance'][0] <= dominance <= mapping['dominance'][1]

    return valence_match and arousal_match and dominance_match


def calculate_seed_match_score(song_seeds: List[str], target_seeds: List[str]) -> float:
    """
    Calculate overlap between song seeds and target emotion seeds.

    Args:
        song_seeds: List of emotion tags from the song
        target_seeds: List of target emotion keywords

    Returns:
        Match score (0.0 to 1.0), representing the proportion of target seeds found
    """
    if not song_seeds or not target_seeds:
        return 0.0

    # Convert to lowercase for case-insensitive matching
    song_seeds_lower = [s.lower() for s in song_seeds]
    target_seeds_lower = [s.lower() for s in target_seeds]

    matches = len(set(song_seeds_lower) & set(target_seeds_lower))
    return matches / len(target_seeds_lower)
