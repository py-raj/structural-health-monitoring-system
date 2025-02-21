# Load trained model
model = tf.keras.models.load_model("crack_detection_model.h5")

def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = img.reshape(1, img_size, img_size, 1)

    prediction = model.predict(img)
    label = categories[np.argmax(prediction)]
    print(f"Predicted Category: {label}")

# Test with an example image
predict_image("test_image.jpg")