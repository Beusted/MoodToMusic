#!/usr/bin/env python3
"""
Real-time Facial Emotion Detection using OpenCV Haar Cascade
This version avoids macOS threading issues by using OpenCV's built-in face detection
Press 'q' or ESC to quit
"""

import os
import sys

# Fix for macOS threading issues - MUST be before importing anything else
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2

# Import mood-to-music components
from emotion_logger import EmotionLogger
from music_recommender import MusicRecommender, print_recommendation

# Emotion labels and colors (BGR format for OpenCV)
emotions = {
    0: {"emotion": "Angry", "color": (42, 69, 193)},
    1: {"emotion": "Disgust", "color": (49, 175, 164)},
    2: {"emotion": "Fear", "color": (155, 52, 40)},
    3: {"emotion": "Happy", "color": (28, 164, 23)},
    4: {"emotion": "Sad", "color": (23, 93, 164)},
    5: {"emotion": "Surprise", "color": (97, 229, 218)},
    6: {"emotion": "Neutral", "color": (200, 72, 108)}
}


def preprocess_face(face_img, manual_scaling=False):
    """
    Preprocess face image for emotion classification.
    Applies manual scaling only if the model doesn't have a built-in Rescaling layer.
    """
    face_img = face_img.astype('float32')

    # Apply legacy scaling for models that require it
    if manual_scaling:
        face_img = face_img / 255.0
        face_img = (face_img - 0.5) * 2.0

    # Always expand dimensions for model input
    face_img = np.expand_dims(face_img, 0)
    face_img = np.expand_dims(face_img, -1)
    return face_img


def main():
    print("=" * 60)
    print("FACIAL EMOTION DETECTION SYSTEM")
    print("=" * 60)

    # Load Haar Cascade for face detection
    print("\n[1/3] Loading face detection model...")
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if face_cascade.empty():
        print(f"ERROR: Could not load Haar Cascade from {face_cascade_path}")
        sys.exit(1)

    print(" Face detection model loaded (Haar Cascade)")

    # --- Smart Model Loading ---
    # Load a model and automatically determine its preprocessing needs.
    # The best model is the default, but you can change this path.
    print("\n[2/3] Loading emotion classification model...")
    emotionModelPath = 'models/emotion_detection/custom_model_weighted.h5'
    # To test other models, change the path above, e.g., to:
    # emotionModelPath = 'models/emotion_detection/emotionModel.hdf5'
    # emotionModelPath = 'models/emotion_detection/custom_model.h5'

    if not os.path.exists(emotionModelPath):
        print(f"ERROR: Emotion model not found at {emotionModelPath}")
        print("Please ensure you have trained the model by running train.py")
        sys.exit(1)

    emotionClassifier = load_model(emotionModelPath, compile=False)
    emotionTargetSize = emotionClassifier.input_shape[1:3]
    
    # Check if the model has a built-in rescaling layer
    needs_manual_scaling = not isinstance(emotionClassifier.layers[0], tf.keras.layers.Rescaling)
    
    print(f" Emotion model loaded: {os.path.basename(emotionModelPath)}")
    print(f"  - Input size: {emotionTargetSize}")
    print(f"  - Needs manual preprocessing: {needs_manual_scaling}")

    # Initialize emotion logger and music recommender
    print("\n[Bonus] Initializing mood-to-music recommendation system...")
    emotion_logger = EmotionLogger(log_interval=1.0, aggregation_period=60.0)
    music_recommender = MusicRecommender(csv_path='muse_v3.csv', models_dir='models/')

    # Initialize webcam
    print("\n[3/3] Initializing webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        sys.exit(1)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(" Webcam initialized")

    # Create display window
    window_name = "Emotion Recognition - Press 'q' or ESC to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 60)
    print("SYSTEM READY - Webcam is now active")
    print("=" * 60)
    print("\nControls:")
    print("  - Press 'q' or ESC to quit")
    print("  - Press 's' to save current frame")
    print("\nDetectable emotions:")
    for emo_id, emo_data in emotions.items():
        print(f"  - {emo_data['emotion']}")
    print("\n")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()

            # Flip frame horizontally to mirror the camera
            frame = cv2.flip(frame, 1)

            if not ret:
                print("WARNING: Failed to grab frame")
                break

            frame_count += 1

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using Haar Cascade
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # If multiple faces, sort by area and track the largest one for mood logging
            if len(faces) > 1:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)

            # Process each detected face
            for face_idx, (x, y, w, h) in enumerate(faces):
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]

                # Skip if face is too small
                if face_roi.size == 0:
                    continue

                try:
                    # Resize face to model input size
                    face_roi = cv2.resize(face_roi, emotionTargetSize)
                except:
                    continue

                # Preprocess face for emotion classification, applying scaling if needed
                processedFace = preprocess_face(face_roi, manual_scaling=needs_manual_scaling)

                # Predict emotion
                emotion_prediction = emotionClassifier.predict(processedFace, verbose=0)
                emotion_probability = np.max(emotion_prediction)

                # Only show emotion if confidence is high enough
                if emotion_probability > 0.36:
                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_name = emotions[emotion_label_arg]['emotion']
                    color = emotions[emotion_label_arg]['color']

                    # Update emotion logger (only for the largest face)
                    if face_idx == 0:
                        emotion_logger.update_current_emotion(emotion_label_arg, emotion_probability)

                    # Draw bounding box around face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Create label background
                    label_text = f"{emotion_name} ({emotion_probability*100:.1f}%)"
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                    # Draw label background
                    cv2.rectangle(frame,
                                (x, y - label_size[1] - 10),
                                (x + label_size[0] + 10, y),
                                color, -1)

                    # Draw label text
                    cv2.putText(frame, label_text,
                              (x + 5, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                              (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    # Low confidence - just draw white box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # Add frame counter and FPS info
            info_text = f"Faces: {len(faces)} | Frame: {frame_count}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Emotion logging (every 1 second)
            if emotion_logger.should_log():
                emotion_logger.log_emotion()

            # Check for 60-second aggregation
            if emotion_logger.should_aggregate():
                prevalent_mood = emotion_logger.get_prevalent_mood()
                if prevalent_mood:
                    print(f"\n{'='*70}")
                    print(f"60-SECOND MOOD SUMMARY")
                    print(f"{'='*70}")
                    print(f"Prevalent Mood: {prevalent_mood['emotion_name']}")
                    print(f"Occurrences: {prevalent_mood['count']}/{prevalent_mood['total_samples']}")
                    print(f"Average Confidence: {prevalent_mood['avg_confidence']*100:.1f}%")
                    print(f"{'='*70}\n")

                    # Get music recommendation
                    recommendation = music_recommender.recommend(prevalent_mood)
                    if recommendation:
                        print_recommendation(recommendation, prevalent_mood)

                emotion_logger.reset_cycle()

            # Display the frame
            cv2.imshow(window_name, frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                print("\nQuitting...")
                break
            elif key == ord('s'):  # 's' to save frame
                filename = f"emotion_capture_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()

        # Force window closure (macOS fix)
        for i in range(5):
            cv2.waitKey(1)

        print(" Cleanup complete")
        print("\nThank you for using Emotion Detection System!")


if __name__ == "__main__":
    main()
