"""
Model Performance Visualization for Presentation

Generates comprehensive graphs showing accuracy and performance metrics
for all 4 ML models: KNN, Decision Trees, K-means Clustering, and SVM.

Usage:
    python visualize_models.py

Output:
    - Saves visualization images to 'visualizations/' directory
    - Creates 8 different graphs for presentation
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score
)
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_data_and_models():
    """Load processed data and trained models."""
    print("Loading data and models...")

    # Load processed songs data
    df = pd.read_csv('data/processed_songs.csv')

    # Parse seeds if needed
    if isinstance(df['seeds'].iloc[0], str):
        df['seeds'] = df['seeds'].apply(lambda x: eval(x) if isinstance(x, str) else [])

    # Load models
    with open('models/music_recommendation/kmeans_music_clusters.pkl', 'rb') as f:
        kmeans_data = pickle.load(f)

    with open('models/music_recommendation/svm_mood_classifier.pkl', 'rb') as f:
        svm_data = pickle.load(f)

    with open('models/music_recommendation/knn_music_index.pkl', 'rb') as f:
        knn_data = pickle.load(f)

    with open('models/music_recommendation/decision_tree_ranker.pkl', 'rb') as f:
        dt_model = pickle.load(f)

    print(f"Loaded {len(df)} songs and all 4 models")

    return df, kmeans_data, svm_data, knn_data, dt_model


def synthesize_mood_labels(df):
    """Recreate mood labels for evaluation (same logic as training)."""
    from mood_mapper import EMOTION_TO_VAD_MAPPING, vad_in_range

    labels = []
    for _, row in df.iterrows():
        v, a, d = row['valence_tags'], row['arousal_tags'], row['dominance_tags']

        best_fit = None
        best_fit_count = 0

        for emotion_id in range(7):
            if vad_in_range(v, a, d, emotion_id):
                best_fit = emotion_id
                best_fit_count += 1

        if best_fit_count == 1:
            labels.append(best_fit)
        else:
            min_distance = float('inf')
            closest_emotion = 6

            for emotion_id, mapping in EMOTION_TO_VAD_MAPPING.items():
                target_v = (mapping['valence'][0] + mapping['valence'][1]) / 2
                target_a = (mapping['arousal'][0] + mapping['arousal'][1]) / 2
                target_d = (mapping['dominance'][0] + mapping['dominance'][1]) / 2

                distance = np.sqrt((v - target_v)**2 + (a - target_a)**2 + (d - target_d)**2)

                if distance < min_distance:
                    min_distance = distance
                    closest_emotion = emotion_id

            labels.append(closest_emotion)

    return np.array(labels)


def create_ranking_labels(df):
    """Recreate ranking labels for decision tree evaluation."""
    from mood_mapper import vad_in_range

    labels = []
    for _, row in df.iterrows():
        v, a, d = row['valence_tags'], row['arousal_tags'], row['dominance_tags']

        fit_count = 0
        for emotion_id in range(7):
            if vad_in_range(v, a, d, emotion_id):
                fit_count += 1

        labels.append(1 if 1 <= fit_count <= 2 else 0)

    return np.array(labels)


def plot_accuracy_comparison(svm_acc, dt_acc, output_dir):
    """Plot 1: Overall accuracy comparison bar chart."""
    print("\n[1/8] Creating accuracy comparison chart...")

    models = ['SVM\n(Mood Classifier)', 'Decision Tree\n(Song Ranker)']
    accuracies = [svm_acc * 100, dt_acc * 100]
    colors = ['#3498db', '#2ecc71']

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Classification Model Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 1_accuracy_comparison.png")


def plot_svm_confusion_matrix(y_true, y_pred, output_dir):
    """Plot 2: SVM confusion matrix."""
    print("\n[2/8] Creating SVM confusion matrix...")

    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_names, yticklabels=emotion_names,
                cbar_kws={'label': 'Number of Songs'}, ax=ax)

    ax.set_xlabel('Predicted Mood', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Mood', fontsize=13, fontweight='bold')
    ax.set_title('SVM Mood Classifier - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_svm_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 2_svm_confusion_matrix.png")


def plot_decision_tree_confusion_matrix(y_true, y_pred, output_dir):
    """Plot 3: Decision Tree confusion matrix."""
    print("\n[3/8] Creating Decision Tree confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Poor Fit', 'Good Fit'],
                yticklabels=['Poor Fit', 'Good Fit'],
                cbar_kws={'label': 'Number of Songs'}, ax=ax)

    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title('Decision Tree Ranker - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_decision_tree_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 3_decision_tree_confusion_matrix.png")


def plot_kmeans_clusters_2d(df, output_dir):
    """Plot 4: K-means clusters visualization (2D: Valence vs Arousal)."""
    print("\n[4/8] Creating K-means 2D cluster visualization...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Sample 5000 points for clearer visualization
    df_sample = df.sample(n=min(5000, len(df)), random_state=42)

    scatter = ax.scatter(df_sample['valence_tags'], df_sample['arousal_tags'],
                        c=df_sample['cluster'], cmap='tab20', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)

    ax.set_xlabel('Valence (Positivity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Arousal (Energy)', fontsize=13, fontweight='bold')
    ax.set_title('K-Means Clustering: Songs by Valence & Arousal\n(12 Emotional Clusters)',
                 fontsize=16, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster ID', fontsize=12, fontweight='bold')

    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_kmeans_2d_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 4_kmeans_2d_clusters.png")


def plot_kmeans_clusters_3d(df, output_dir):
    """Plot 5: K-means clusters visualization (3D: VAD space)."""
    print("\n[5/8] Creating K-means 3D cluster visualization...")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Sample 3000 points for performance
    df_sample = df.sample(n=min(3000, len(df)), random_state=42)

    scatter = ax.scatter(df_sample['valence_tags'],
                        df_sample['arousal_tags'],
                        df_sample['dominance_tags'],
                        c=df_sample['cluster'], cmap='tab20',
                        alpha=0.6, s=20, edgecolors='black', linewidth=0.2)

    ax.set_xlabel('Valence (Positivity)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Arousal (Energy)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Dominance (Power)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('K-Means Clustering in 3D VAD Space\n(Valence-Arousal-Dominance)',
                 fontsize=16, fontweight='bold', pad=30)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Cluster ID', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_kmeans_3d_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 5_kmeans_3d_clusters.png")


def plot_cluster_metrics(df, kmeans_model, output_dir):
    """Plot 6: K-means cluster quality metrics."""
    print("\n[6/8] Creating K-means cluster quality metrics...")

    X = df[['valence_tags', 'arousal_tags', 'dominance_tags']].values
    cluster_labels = df['cluster'].values

    # Calculate metrics
    silhouette = silhouette_score(X, cluster_labels)
    davies_bouldin = davies_bouldin_score(X, cluster_labels)
    inertia = kmeans_model['model'].inertia_

    # Cluster sizes
    cluster_sizes = df['cluster'].value_counts().sort_index()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cluster sizes
    ax1.bar(cluster_sizes.index, cluster_sizes.values, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Cluster ID', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Songs', fontsize=11, fontweight='bold')
    ax1.set_title('Cluster Size Distribution', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Quality metrics
    metrics = ['Silhouette\nScore', 'Davies-Bouldin\nIndex']
    values = [silhouette, davies_bouldin]
    colors_met = ['#2ecc71', '#e74c3c']

    bars = ax2.bar(metrics, values, color=colors_met, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Clustering Quality Metrics', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Inertia
    ax3.bar(['Within-Cluster\nSum of Squares'], [inertia], color='#9b59b6', alpha=0.8, edgecolor='black')
    ax3.text(0, inertia, f'{inertia:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.set_ylabel('WCSS', fontsize=11, fontweight='bold')
    ax3.set_title('K-Means Inertia (Lower is Better)', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Info text
    ax4.axis('off')
    info_text = f"""
    K-Means Clustering Summary
    {'='*40}

    Number of Clusters: 12
    Total Songs: {len(df):,}
    Average Cluster Size: {len(df)//12:,}

    Silhouette Score: {silhouette:.4f}
    (Range: -1 to 1, Higher is better)

    Davies-Bouldin Index: {davies_bouldin:.4f}
    (Lower is better)

    Inertia (WCSS): {inertia:,.0f}
    (Within-Cluster Sum of Squares)
    """
    ax4.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('K-Means Clustering Performance Metrics', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_kmeans_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 6_kmeans_metrics.png")


def plot_mood_distribution(df, output_dir):
    """Plot 7: Mood label distribution across dataset."""
    print("\n[7/8] Creating mood distribution chart...")

    mood_labels = synthesize_mood_labels(df)
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    unique, counts = np.unique(mood_labels, return_counts=True)
    percentages = (counts / len(mood_labels)) * 100

    colors_palette = ['#e74c3c', '#95a5a6', '#34495e', '#f39c12', '#3498db', '#9b59b6', '#2ecc71']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Bar chart
    bars = ax1.bar([emotion_names[i] for i in unique], counts,
                    color=[colors_palette[i] for i in unique],
                    alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Number of Songs', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Mood Category', fontsize=12, fontweight='bold')
    ax1.set_title('Song Distribution by Mood', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Pie chart
    ax2.pie(counts, labels=[emotion_names[i] for i in unique], autopct='%1.1f%%',
            colors=[colors_palette[i] for i in unique], startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Mood Distribution (Percentage)', fontsize=14, fontweight='bold')

    plt.suptitle('Dataset Mood Analysis (SVM Training Data)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_mood_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 7_mood_distribution.png")


def plot_model_pipeline_summary(svm_acc, dt_acc, df, output_dir):
    """Plot 8: Complete pipeline summary with all 4 models."""
    print("\n[8/8] Creating complete pipeline summary...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Model 1: K-means
    ax1.text(0.5, 0.7, 'K-Means Clustering', ha='center', fontsize=18, fontweight='bold')
    ax1.text(0.5, 0.5, f'12 Clusters\n{len(df):,} Songs Organized', ha='center', fontsize=14)
    ax1.text(0.5, 0.3, 'Purpose: Coarse Filtering\n90K songs → ~6K candidates', ha='center', fontsize=12)
    ax1.text(0.5, 0.1, '✓ Unsupervised Learning', ha='center', fontsize=11, style='italic', color='green')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='blue', linewidth=3))

    # Model 2: SVM
    ax2.text(0.5, 0.7, 'SVM Classifier', ha='center', fontsize=18, fontweight='bold')
    ax2.text(0.5, 0.5, f'Accuracy: {svm_acc*100:.2f}%', ha='center', fontsize=16, fontweight='bold', color='darkgreen')
    ax2.text(0.5, 0.3, 'Purpose: Mood Validation\n7 Emotion Categories', ha='center', fontsize=12)
    ax2.text(0.5, 0.1, '✓ RBF Kernel, Probabilistic', ha='center', fontsize=11, style='italic', color='green')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='purple', linewidth=3))

    # Model 3: KNN
    ax3.text(0.5, 0.7, 'K-Nearest Neighbors', ha='center', fontsize=18, fontweight='bold')
    ax3.text(0.5, 0.5, 'k = 15 neighbors', ha='center', fontsize=14)
    ax3.text(0.5, 0.3, 'Purpose: Similarity Search\nFinds similar songs in VAD space', ha='center', fontsize=12)
    ax3.text(0.5, 0.1, '✓ Ball Tree Algorithm', ha='center', fontsize=11, style='italic', color='green')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='orange', linewidth=3))

    # Model 4: Decision Tree
    ax4.text(0.5, 0.7, 'Decision Tree Ranker', ha='center', fontsize=18, fontweight='bold')
    ax4.text(0.5, 0.5, f'Accuracy: {dt_acc*100:.2f}%', ha='center', fontsize=16, fontweight='bold', color='darkgreen')
    ax4.text(0.5, 0.3, 'Purpose: Final Ranking\nSelects best match', ha='center', fontsize=12)
    ax4.text(0.5, 0.1, '✓ Max Depth: 10', ha='center', fontsize=11, style='italic', color='green')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='green', linewidth=3))

    plt.suptitle('4-Stage ML Pipeline Overview', fontsize=20, fontweight='bold', y=0.98)

    # Add pipeline flow arrows
    fig.text(0.5, 0.48, '↓ Pipeline Flow ↓', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '8_pipeline_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 8_pipeline_summary.png")


def main():
    """Main visualization generation function."""
    print("="*70)
    print("MODEL PERFORMANCE VISUALIZATION FOR PRESENTATION")
    print("="*70)

    # Create output directory
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")

    # Load data and models
    df, kmeans_data, svm_data, knn_data, dt_model = load_data_and_models()

    # Evaluate SVM
    print("\nEvaluating SVM model...")
    df['mood_label'] = synthesize_mood_labels(df)
    X_svm = df[['valence_tags', 'arousal_tags', 'dominance_tags', 'genre_encoded']].values
    y_svm = df['mood_label'].values

    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
        X_svm, y_svm, test_size=0.2, random_state=42, stratify=y_svm
    )

    X_test_svm_scaled = svm_data['scaler'].transform(X_test_svm)
    y_pred_svm = svm_data['model'].predict(X_test_svm_scaled)
    svm_accuracy = accuracy_score(y_test_svm, y_pred_svm)
    print(f"  SVM Test Accuracy: {svm_accuracy*100:.2f}%")

    # Evaluate Decision Tree
    print("\nEvaluating Decision Tree model...")
    df['fit_label'] = create_ranking_labels(df)
    X_dt = df[['valence_tags', 'arousal_tags', 'dominance_tags', 'genre_encoded']].values
    y_dt = df['fit_label'].values

    X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
        X_dt, y_dt, test_size=0.2, random_state=42
    )

    y_pred_dt = dt_model.predict(X_test_dt)
    dt_accuracy = accuracy_score(y_test_dt, y_pred_dt)
    print(f"  Decision Tree Test Accuracy: {dt_accuracy*100:.2f}%")

    # Generate all visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    plot_accuracy_comparison(svm_accuracy, dt_accuracy, output_dir)
    plot_svm_confusion_matrix(y_test_svm, y_pred_svm, output_dir)
    plot_decision_tree_confusion_matrix(y_test_dt, y_pred_dt, output_dir)
    plot_kmeans_clusters_2d(df, output_dir)
    plot_kmeans_clusters_3d(df, output_dir)
    plot_cluster_metrics(df, kmeans_data, output_dir)
    plot_mood_distribution(df, output_dir)
    plot_model_pipeline_summary(svm_accuracy, dt_accuracy, df, output_dir)

    # Summary
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated 8 visualization images in: {output_dir}/")
    print("\nFiles created:")
    for i in range(1, 9):
        files = [f for f in os.listdir(output_dir) if f.startswith(f'{i}_')]
        if files:
            print(f"  {i}. {files[0]}")

    print("\nThese visualizations are ready for your presentation!")
    print("\nModel Performance Summary:")
    print(f"  • SVM Accuracy: {svm_accuracy*100:.2f}%")
    print(f"  • Decision Tree Accuracy: {dt_accuracy*100:.2f}%")
    print(f"  • K-means Clusters: 12")
    print(f"  • KNN Neighbors: 15")
    print(f"  • Total Songs Analyzed: {len(df):,}")


if __name__ == '__main__':
    # Check if required packages are installed
    try:
        import matplotlib
        import seaborn
    except ImportError:
        print("ERROR: Required packages not installed.")
        print("Please run: pip install matplotlib seaborn")
        sys.exit(1)

    main()
