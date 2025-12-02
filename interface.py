import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("crack_detection_model.h5")

# Define categories (same as training)
categories = ["Healthy", "Damaged"]

# Define image size (same as used during training)
img_size = 128  

def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    img = cv2.resize(img, (img_size, img_size))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values (0-1)
    img = img.reshape(1, img_size, img_size, 1)  # Reshape for CNN

    prediction = model.predict(img)
    label = categories[np.argmax(prediction)]  # Get predicted class
    print(f"Predicted Category: {label}")

# Test with an example image
predict_image(r"D:\dataset\Healthy\20000.jpg")  # Replace with your image path
