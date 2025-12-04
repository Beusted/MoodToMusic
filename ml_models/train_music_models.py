"""
Music Model Training Pipeline

Trains all 4 ML models for the mood-to-music recommendation system:
1. K-means clustering (for coarse song filtering)
2. SVM classification (for mood category validation)
3. KNN index (for finding similar songs)
4. Decision Tree (for final ranking)

Usage:
    python ml_models/train_music_models.py
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

# Add parent directory to path to import mood_mapper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mood_mapper import EMOTION_TO_VAD_MAPPING, vad_in_range

warnings.filterwarnings('ignore')


def safe_parse_seeds(x):
    """Safely parse the seeds column (list stored as string)."""
    if pd.isna(x):
        return []
    try:
        if isinstance(x, str):
            return eval(x)
        elif isinstance(x, list):
            return x
        else:
            return []
    except Exception:
        return []


def preprocess_music_data(csv_path):
    """
    Preprocess the music dataset.

    Args:
        csv_path: Path to muse_v3.csv

    Returns:
        Preprocessed DataFrame
    """
    print("Loading music dataset...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} songs")

    # Parse seeds column
    print("Parsing emotion seeds...")
    df['seeds'] = df['seeds'].apply(safe_parse_seeds)

    # Handle missing values
    print("Handling missing values...")
    original_count = len(df)
    df = df.dropna(subset=['valence_tags', 'arousal_tags', 'dominance_tags'])
    print(f"Removed {original_count - len(df)} songs with missing VAD values")

    # Filter invalid VAD values (should be 0-10 range)
    print("Filtering invalid VAD values...")
    df = df[
        (df['valence_tags'] >= 0) & (df['valence_tags'] <= 10) &
        (df['arousal_tags'] >= 0) & (df['arousal_tags'] <= 10) &
        (df['dominance_tags'] >= 0) & (df['dominance_tags'] <= 10)
    ]
    print(f"Remaining songs: {len(df)}")

    # Encode genres
    print("Encoding genres...")
    df['genre'] = df['genre'].fillna('unknown')
    le = LabelEncoder()
    df['genre_encoded'] = le.fit_transform(df['genre'])

    # Save label encoder for later use
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'music_recommendation')
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'genre_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    print(f"Encoded {len(le.classes_)} unique genres")

    return df


def synthesize_mood_labels(df):
    """
    Create mood labels (0-6) from VAD values based on EMOTION_TO_VAD_MAPPING.

    Args:
        df: DataFrame with valence_tags, arousal_tags, dominance_tags columns

    Returns:
        Array of mood labels
    """
    print("Synthesizing mood labels from VAD values...")
    labels = []

    for _, row in df.iterrows():
        v, a, d = row['valence_tags'], row['arousal_tags'], row['dominance_tags']

        # Check which emotion range this song best fits
        best_fit = None
        best_fit_count = 0

        # Try to match to each emotion's VAD ranges
        for emotion_id in range(7):
            if vad_in_range(v, a, d, emotion_id):
                best_fit = emotion_id
                best_fit_count += 1

        # If matches exactly one emotion, use it
        if best_fit_count == 1:
            labels.append(best_fit)
        # Otherwise, use distance-based assignment
        else:
            # Calculate Euclidean distance to each emotion's midpoint
            min_distance = float('inf')
            closest_emotion = 6  # Default to Neutral

            for emotion_id, mapping in EMOTION_TO_VAD_MAPPING.items():
                target_v = (mapping['valence'][0] + mapping['valence'][1]) / 2
                target_a = (mapping['arousal'][0] + mapping['arousal'][1]) / 2
                target_d = (mapping['dominance'][0] + mapping['dominance'][1]) / 2

                distance = np.sqrt((v - target_v)**2 + (a - target_a)**2 + (d - target_d)**2)

                if distance < min_distance:
                    min_distance = distance
                    closest_emotion = emotion_id

            labels.append(closest_emotion)

    # Print distribution
    unique, counts = np.unique(labels, return_counts=True)
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    print("\nMood label distribution:")
    for emotion_id, count in zip(unique, counts):
        print(f"  {emotion_names[emotion_id]}: {count} songs ({count/len(labels)*100:.1f}%)")

    return np.array(labels)


def train_kmeans(df, n_clusters=12):
    """
    Train K-means clustering model.

    Args:
        df: DataFrame with VAD features
        n_clusters: Number of clusters (default: 12)

    Returns:
        Trained KMeans model
    """
    print(f"\nTraining K-means clustering (n_clusters={n_clusters})...")

    # Prepare features
    X = df[['valence_tags', 'arousal_tags', 'dominance_tags']].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train K-means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300,
        verbose=0
    )
    kmeans.fit(X_scaled)

    # Add cluster assignments to dataframe
    df['cluster'] = kmeans.predict(X_scaled)

    # Print cluster sizes
    print("\nCluster sizes:")
    for cluster_id in range(n_clusters):
        count = (df['cluster'] == cluster_id).sum()
        print(f"  Cluster {cluster_id}: {count} songs")

    # Save model and scaler
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'music_recommendation')
    with open(os.path.join(models_dir, 'kmeans_music_clusters.pkl'), 'wb') as f:
        pickle.dump({'model': kmeans, 'scaler': scaler}, f)
    print(f"Saved K-means model to models/kmeans_music_clusters.pkl")

    return kmeans


def train_svm(df):
    """
    Train SVM mood classifier.

    Args:
        df: DataFrame with VAD features and mood labels

    Returns:
        Trained SVM model
    """
    print("\nTraining SVM mood classifier...")

    # Synthesize mood labels
    df['mood_label'] = synthesize_mood_labels(df)

    # Prepare features: VAD + genre
    X = df[['valence_tags', 'arousal_tags', 'dominance_tags', 'genre_encoded']].values
    y = df['mood_label'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    print("Training SVM (this may take a few minutes)...")
    svm = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,  # For confidence scores
        random_state=42,
        verbose=False
    )
    svm.fit(X_train_scaled, y_train)

    # Evaluate
    train_accuracy = svm.score(X_train_scaled, y_train)
    test_accuracy = svm.score(X_test_scaled, y_test)
    print(f"SVM Training Accuracy: {train_accuracy*100:.2f}%")
    print(f"SVM Test Accuracy: {test_accuracy*100:.2f}%")

    # Save model and scaler
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'music_recommendation')
    with open(os.path.join(models_dir, 'svm_mood_classifier.pkl'), 'wb') as f:
        pickle.dump({'model': svm, 'scaler': scaler}, f)
    print(f"Saved SVM model to models/svm_mood_classifier.pkl")

    return svm


def build_knn_index(df):
    """
    Build KNN index for similarity search.

    Args:
        df: DataFrame with VAD features

    Returns:
        KNN model
    """
    print("\nBuilding KNN index...")

    # Prepare features: VAD only
    X = df[['valence_tags', 'arousal_tags', 'dominance_tags']].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build KNN index
    knn = NearestNeighbors(
        n_neighbors=20,
        algorithm='ball_tree',  # Faster for 3D space
        metric='euclidean'
    )
    knn.fit(X_scaled)

    # Save model, scaler, and indices
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'music_recommendation')
    with open(os.path.join(models_dir, 'knn_music_index.pkl'), 'wb') as f:
        pickle.dump({
            'model': knn,
            'scaler': scaler,
            'song_indices': df.index.tolist()
        }, f)
    print(f"Saved KNN index to models/knn_music_index.pkl")

    return knn


def create_ranking_labels(df):
    """
    Create binary labels for decision tree: good_fit (1) vs poor_fit (0).

    A song is a "good fit" if it fits 1-2 emotions well (focused emotional profile).
    A song is a "poor fit" if it fits 0 or 3+ emotions (too generic or unclear).

    Args:
        df: DataFrame with VAD features

    Returns:
        Array of binary labels
    """
    print("Creating ranking labels...")
    labels = []

    for _, row in df.iterrows():
        v, a, d = row['valence_tags'], row['arousal_tags'], row['dominance_tags']

        # Check how many emotion ranges this song fits
        fit_count = 0
        for emotion_id in range(7):
            if vad_in_range(v, a, d, emotion_id):
                fit_count += 1

        # Good fit: 1-2 moods, Poor fit: 0 or 3+ moods
        labels.append(1 if 1 <= fit_count <= 2 else 0)

    labels = np.array(labels)
    good_fit_count = labels.sum()
    poor_fit_count = len(labels) - good_fit_count
    print(f"  Good fit songs: {good_fit_count} ({good_fit_count/len(labels)*100:.1f}%)")
    print(f"  Poor fit songs: {poor_fit_count} ({poor_fit_count/len(labels)*100:.1f}%)")

    return labels


def train_decision_tree(df):
    """
    Train decision tree for ranking candidates.

    Args:
        df: DataFrame with VAD features

    Returns:
        Trained DecisionTreeClassifier
    """
    print("\nTraining Decision Tree ranker...")

    # Create labels
    df['fit_label'] = create_ranking_labels(df)

    # Prepare features
    X = df[['valence_tags', 'arousal_tags', 'dominance_tags', 'genre_encoded']].values
    y = df['fit_label'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train decision tree
    dt = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=100,
        min_samples_leaf=50,
        random_state=42
    )
    dt.fit(X_train, y_train)

    # Evaluate
    train_accuracy = dt.score(X_train, y_train)
    test_accuracy = dt.score(X_test, y_test)
    print(f"Decision Tree Training Accuracy: {train_accuracy*100:.2f}%")
    print(f"Decision Tree Test Accuracy: {test_accuracy*100:.2f}%")

    # Save model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'music_recommendation')
    with open(os.path.join(models_dir, 'decision_tree_ranker.pkl'), 'wb') as f:
        pickle.dump(dt, f)
    print(f"Saved Decision Tree model to models/decision_tree_ranker.pkl")

    return dt


def main():
    """Main training pipeline."""
    print("="*70)
    print("MUSIC RECOMMENDATION SYSTEM - MODEL TRAINING PIPELINE")
    print("="*70)

    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    csv_path = os.path.join(project_dir, 'muse_v3.csv')
    data_dir = os.path.join(project_dir, 'data')
    models_dir = os.path.join(project_dir, 'models')

    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, 'music_recommendation'), exist_ok=True)
    os.makedirs(os.path.join(models_dir, 'emotion_detection'), exist_ok=True)

    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find muse_v3.csv at {csv_path}")
        return

    # Step 1: Preprocess data
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    df = preprocess_music_data(csv_path)

    # Step 2: Train K-means
    print("\n" + "="*70)
    print("STEP 2: K-MEANS CLUSTERING")
    print("="*70)
    kmeans = train_kmeans(df, n_clusters=12)

    # Step 3: Train SVM
    print("\n" + "="*70)
    print("STEP 3: SVM CLASSIFICATION")
    print("="*70)
    svm = train_svm(df)

    # Step 4: Build KNN index
    print("\n" + "="*70)
    print("STEP 4: KNN INDEX BUILDING")
    print("="*70)
    knn = build_knn_index(df)

    # Step 5: Train Decision Tree
    print("\n" + "="*70)
    print("STEP 5: DECISION TREE TRAINING")
    print("="*70)
    dt = train_decision_tree(df)

    # Step 6: Save processed data
    print("\n" + "="*70)
    print("STEP 6: SAVING PROCESSED DATA")
    print("="*70)
    processed_csv_path = os.path.join(data_dir, 'processed_songs.csv')
    df.to_csv(processed_csv_path, index=False)
    print(f"Saved processed dataset to {processed_csv_path}")
    print(f"Dataset contains {len(df)} songs with {len(df.columns)} columns")

    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  1. models/music_recommendation/kmeans_music_clusters.pkl")
    print(f"  2. models/music_recommendation/svm_mood_classifier.pkl")
    print(f"  3. models/music_recommendation/knn_music_index.pkl")
    print(f"  4. models/music_recommendation/decision_tree_ranker.pkl")
    print(f"  5. models/music_recommendation/genre_encoder.pkl")
    print(f"  6. data/processed_songs.csv")
    print("\nYou can now run the emotion detection system with music recommendations!")


if __name__ == '__main__':
    main()
