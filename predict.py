import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("models/medicine_model.h5")

# Class names (must match training order)
class_names = ['aceta', 'alatrol', 'esonix']

# Load test image
img_path = "test.jpg"   # put your test image here
img = cv2.imread(img_path)

# Resize same as training
img = cv2.resize(img, (128, 128))

# Normalize
img = img / 255.0

# Expand dimensions
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print("Prediction:", predicted_class)
print("Confidence:", confidence)