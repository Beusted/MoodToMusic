# MoodToMusic

Using facial emotion recognition to recommend the next song for your mood.

## Features

- Real-time facial emotion detection using webcam
- Detects 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Built with OpenCV and Keras/TensorFlow
- Live confidence scores and visual feedback

## Setup

### 1. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
### 2.5. (Optional Step) Train the model

```bash
python3 train.py
```

### 3. Run the application

```bash
python3 emotion.py
```

## Usage

Once running:
- **Press 'q' or ESC** to quit
- **Press 's'** to save current frame

## Model

The emotion classification model (`models/emotionModel.hdf5`) is a pre-trained CNN that:
- Takes grayscale 48x48 or 64x64 face images as input
- Outputs probabilities for 7 emotion classes
- Only displays predictions with >36% confidence

## Requirements

- Python 3.x
- OpenCV
- Keras/TensorFlow
- NumPy
