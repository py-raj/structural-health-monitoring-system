import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load dataset (Replace with actual dataset path)
dataset_path = r"D:\dataset"
categories = ["Healthy", "Damaged"]

# Image processing parameters
img_size = 128
data = []
labels = []

# Load and preprocess images
for category in categories:
    path = os.path.join(dataset_path, category)
    label = categories.index(category)
    for img_name in os.listdir(path):
        try:
            img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            data.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image: {img_name}")

# Convert to NumPy arrays and normalize
data = np.array(data) / 255.0
data = data.reshape(-1, img_size, img_size, 1)
labels = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 output classes (Healthy, Damaged)
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save trained model
model.save("crack_detection_model.h5")

print("Model training complete and saved!")