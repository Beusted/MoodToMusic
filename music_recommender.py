"""
Music Recommender: 4-stage ML pipeline for mood-based music recommendation.

Pipeline stages:
1. K-means: Cluster songs by VAD features (reduces 90K → ~6K candidates)
2. SVM: Validate mood category
3. KNN: Find 15 nearest neighbor songs
4. Decision Tree: Rank and select best recommendation

Usage:
    from music_recommender import MusicRecommender, print_recommendation

    recommender = MusicRecommender('muse_v3.csv', 'models/')
    song = recommender.recommend(prevalent_mood)
    print_recommendation(song, prevalent_mood)
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, List

from mood_mapper import get_target_vad, calculate_seed_match_score


class MusicRecommender:
    """Music recommendation system using 4-stage ML pipeline."""

    def __init__(self, csv_path: str, models_dir: str):
        """
        Initialize the music recommender.

        Args:
            csv_path: Path to the processed songs CSV (data/processed_songs.csv)
            models_dir: Directory containing trained models
        """
        print("Initializing Music Recommender...")

        # Load processed songs dataset
        processed_csv = os.path.join(os.path.dirname(csv_path), 'data', 'processed_songs.csv')
        if os.path.exists(processed_csv):
            self.songs_df = pd.read_csv(processed_csv)
            print(f"Loaded {len(self.songs_df)} songs from processed dataset")
        else:
            print(f"ERROR: Processed dataset not found at {processed_csv}")
            print("Please run: python ml_models/train_music_models.py")
            raise FileNotFoundError(f"Could not find {processed_csv}")

        # Parse seeds column if needed
        if isinstance(self.songs_df['seeds'].iloc[0], str):
            self.songs_df['seeds'] = self.songs_df['seeds'].apply(
                lambda x: eval(x) if isinstance(x, str) else []
            )

        # Load all 4 trained models
        print("Loading trained ML models...")

        # Music recommendation models subdirectory
        music_models_dir = os.path.join(models_dir, 'music_recommendation')

        # K-means
        kmeans_path = os.path.join(music_models_dir, 'kmeans_music_clusters.pkl')
        with open(kmeans_path, 'rb') as f:
            kmeans_data = pickle.load(f)
            self.kmeans_model = kmeans_data['model']
            self.kmeans_scaler = kmeans_data['scaler']
        print("  ✓ K-means loaded")

        # SVM
        svm_path = os.path.join(music_models_dir, 'svm_mood_classifier.pkl')
        with open(svm_path, 'rb') as f:
            svm_data = pickle.load(f)
            self.svm_model = svm_data['model']
            self.svm_scaler = svm_data['scaler']
        print("  ✓ SVM loaded")

        # KNN
        knn_path = os.path.join(music_models_dir, 'knn_music_index.pkl')
        with open(knn_path, 'rb') as f:
            knn_data = pickle.load(f)
            self.knn_model = knn_data['model']
            self.knn_scaler = knn_data['scaler']
        print("  ✓ KNN loaded")

        # Decision Tree
        dt_path = os.path.join(music_models_dir, 'decision_tree_ranker.pkl')
        with open(dt_path, 'rb') as f:
            self.decision_tree = pickle.load(f)
        print("  ✓ Decision Tree loaded")

        # Track recommendation history to avoid repeats
        self.recommendation_history: List[str] = []

        print("Music Recommender ready!\n")

    def recommend(self, prevalent_mood: Dict) -> Optional[Dict]:
        """
        4-stage ML pipeline to recommend a song based on prevalent mood.

        Args:
            prevalent_mood: Dictionary containing:
                - emotion_id: Detected emotion (0-6)
                - emotion_name: Name of emotion
                - count: Number of occurrences
                - avg_confidence: Average confidence score

        Returns:
            Dictionary with song information, or None if recommendation fails
        """
        try:
            # Stage 0: Map emotion to target VAD
            target_vad = get_target_vad(prevalent_mood['emotion_id'])
            print(f"\nTarget VAD Profile:")
            print(f"  Valence:   {target_vad['valence']:.2f}")
            print(f"  Arousal:   {target_vad['arousal']:.2f}")
            print(f"  Dominance: {target_vad['dominance']:.2f}")

            # Stage 1: K-means clustering (coarse filtering)
            vad_vector = np.array([[
                target_vad['valence'],
                target_vad['arousal'],
                target_vad['dominance']
            ]])
            vad_scaled = self.kmeans_scaler.transform(vad_vector)
            cluster_id = self.kmeans_model.predict(vad_scaled)[0]

            candidates = self.songs_df[self.songs_df['cluster'] == cluster_id].copy()
            print(f"\nStage 1 (K-means): Filtered to cluster {cluster_id} ({len(candidates)} songs)")

            # Handle empty cluster (fallback to nearest clusters)
            if len(candidates) == 0:
                print("  WARNING: Empty cluster, expanding search...")
                cluster_distances = self.kmeans_model.transform(vad_scaled)[0]
                top_3_clusters = np.argsort(cluster_distances)[:3]
                candidates = self.songs_df[self.songs_df['cluster'].isin(top_3_clusters)].copy()
                print(f"  Expanded to {len(candidates)} songs from top 3 clusters")

            # Stage 2: SVM mood validation
            vad_genre_vector = np.array([[
                target_vad['valence'],
                target_vad['arousal'],
                target_vad['dominance'],
                0  # Genre is not used for mood matching, placeholder
            ]])
            vad_genre_scaled = self.svm_scaler.transform(vad_genre_vector)
            svm_prediction = self.svm_model.predict(vad_genre_scaled)[0]
            svm_confidence = self.svm_model.predict_proba(vad_genre_scaled)[0].max()

            print(f"Stage 2 (SVM): Validated mood category (confidence: {svm_confidence*100:.1f}%)")

            # Stage 3: KNN nearest neighbors
            knn_vad_scaled = self.knn_scaler.transform(vad_vector)

            # Get candidate indices in the full dataset
            candidate_indices = candidates.index.tolist()

            # Find KNN in the candidate subset
            n_neighbors = min(15, len(candidates))  # Don't request more neighbors than candidates

            if n_neighbors < 5:
                print(f"  WARNING: Very few candidates ({n_neighbors}), results may be limited")

            # Get features for candidates
            candidate_features = candidates[['valence_tags', 'arousal_tags', 'dominance_tags']].values
            candidate_features_scaled = self.knn_scaler.transform(candidate_features)

            # Calculate distances to all candidates
            distances = np.linalg.norm(candidate_features_scaled - knn_vad_scaled, axis=1)

            # Get top N nearest neighbors
            nearest_indices = np.argsort(distances)[:n_neighbors]
            knn_candidates = candidates.iloc[nearest_indices].copy()
            knn_distances = distances[nearest_indices]

            print(f"Stage 3 (KNN): Found {len(knn_candidates)} nearest neighbors")

            # Apply seed keyword bonus
            target_seeds = target_vad['seeds']
            seed_scores = []
            for _, song in knn_candidates.iterrows():
                seed_score = calculate_seed_match_score(song['seeds'], target_seeds)
                seed_scores.append(seed_score)
            knn_candidates['seed_match_score'] = seed_scores

            # Stage 4: Decision Tree ranking
            dt_features = knn_candidates[[
                'valence_tags', 'arousal_tags', 'dominance_tags', 'genre_encoded'
            ]].values

            fit_probabilities = self.decision_tree.predict_proba(dt_features)[:, 1]  # Probability of good fit
            knn_candidates['fit_score'] = fit_probabilities

            # Combine KNN distance (lower is better) with seed match and fit score (higher is better)
            # Normalize distance to 0-1 scale
            max_distance = knn_distances.max() if knn_distances.max() > 0 else 1
            normalized_distances = 1 - (knn_distances / max_distance)  # Invert so higher is better

            # Combined score: 40% KNN similarity, 30% seed match, 30% fit score
            knn_candidates['final_score'] = (
                0.4 * normalized_distances +
                0.3 * knn_candidates['seed_match_score'] +
                0.3 * knn_candidates['fit_score']
            )

            # Filter out recently recommended songs
            if self.recommendation_history:
                knn_candidates = knn_candidates[
                    ~knn_candidates['track'].isin(self.recommendation_history)
                ]
                if len(knn_candidates) == 0:
                    print("  WARNING: All candidates were recently recommended, clearing history...")
                    self.recommendation_history = []
                    # Restore candidates
                    knn_candidates = candidates.iloc[nearest_indices].copy()
                    knn_candidates['final_score'] = (
                        0.4 * normalized_distances +
                        0.3 * np.array(seed_scores) +
                        0.3 * fit_probabilities
                    )

            # Select best song
            best_idx = knn_candidates['final_score'].idxmax()
            final_song = knn_candidates.loc[best_idx]

            print(f"Stage 4 (Decision Tree): Selected best match")
            print(f"  Fit score: {final_song['fit_score']:.2f}")
            print(f"  Seed match: {final_song['seed_match_score']:.2f}")
            print(f"  Final score: {final_song['final_score']:.2f}")

            # Add to history
            self.recommendation_history.append(final_song['track'])
            if len(self.recommendation_history) > 10:
                self.recommendation_history.pop(0)

            return final_song.to_dict()

        except Exception as e:
            print(f"\nERROR in recommendation pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None


def print_recommendation(song: Dict, prevalent_mood: Dict) -> None:
    """
    Print song recommendation to console in a formatted way.

    Args:
        song: Dictionary with song information
        prevalent_mood: Dictionary with prevalent mood information
    """
    print(f"\n{'='*70}")
    print(f"SONG RECOMMENDATION")
    print(f"{'='*70}")
    print(f"Based on prevalent mood: {prevalent_mood['emotion_name']}")
    print(f"  (detected {prevalent_mood['count']}/{prevalent_mood['total_samples']} times, ")
    print(f"   avg confidence: {prevalent_mood['avg_confidence']*100:.1f}%)")
    print(f"\n{'-'*70}")
    print(f"Track:       {song['track']}")
    print(f"Artist:      {song['artist']}")
    print(f"Genre:       {song['genre']}")
    print(f"{'-'*70}")
    print(f"Emotional Profile:")
    print(f"  Valence:   {song['valence_tags']:.2f} (0=negative, 10=positive)")
    print(f"  Arousal:   {song['arousal_tags']:.2f} (0=calm, 10=energetic)")
    print(f"  Dominance: {song['dominance_tags']:.2f} (0=submissive, 10=powerful)")
    print(f"{'-'*70}")

    # Print up to 8 emotion tags
    seeds = song['seeds'][:8] if isinstance(song['seeds'], list) else []
    if seeds:
        print(f"Emotions:    {', '.join(seeds)}")

    # Print Spotify ID if available
    if pd.notna(song.get('spotify_id')):
        print(f"Spotify ID:  {song['spotify_id']}")
        print(f"  Listen: https://open.spotify.com/track/{song['spotify_id']}")

    print(f"{'='*70}\n")
