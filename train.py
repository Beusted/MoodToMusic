
import os

# Fix for macOS threading issues - MUST be before importing tensorflow
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# --- Configuration ---
# Data paths
TRAIN_DATA_PATH = "data/train"
TEST_DATA_PATH = "data/test"

# Model and image parameters
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64 # Larger batch size can speed up training
COLOR_MODE = "grayscale"
EPOCHS = 30 # Start with 30 and see how the accuracy improves

# --- 1. Load Data ---
print("Loading datasets...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATA_PATH,
    validation_split=0.2, # Use part of the training data for validation
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATA_PATH, # Create validation set from training data
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
)

# Test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DATA_PATH,
    image_size=IMAGE_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
)


CLASS_NAMES = train_dataset.class_names
NUM_CLASSES = len(CLASS_NAMES)
print(f"Found {NUM_CLASSES} classes: {CLASS_NAMES}")

# --- 2. Calculate Class Weights for Imbalanced Data ---
print("Calculating class weights to handle data imbalance...")
# Extract all labels from the training dataset
train_labels = np.concatenate([y for x, y in train_dataset], axis=0)

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights calculated:")
for i, weight in class_weight_dict.items():
    print(f"  - Class '{CLASS_NAMES[i]}': {weight:.2f}")


# --- 3. Define the CNN Model Architecture ---
print("Building the model...")

model = models.Sequential([
    # Normalize pixel values from [0, 255] to [0, 1]
    layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),

    # Convolutional Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    # Convolutional Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    # Convolutional Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    # Flatten the feature maps and feed into dense layers
    layers.Flatten(),

    # Fully Connected Dense Layers
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax') # Softmax for multi-class probability
])

model.summary()


# --- 4. Compile the Model ---
print("Compiling the model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# --- 5. Train the Model ---
print("Starting training with class weights...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weight_dict
)
print("Training complete.")


# --- 6. Evaluate the Model ---
print("Evaluating model performance on the test set...")
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy*100:.2f}%")


# --- 7. Save the Trained Model ---
# Save the final model in the models directory
SAVE_PATH = 'models/custom_model_weighted.h5'
model.save(SAVE_PATH)
print(f"Model saved successfully to {SAVE_PATH}")
