import tensorflow as tf
from tensorflow.keras import layers, models

# Image settings
IMG_SIZE = 128
BATCH_SIZE = 8

# Load dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Get class names
class_names = train_data.class_names
print("Classes:", class_names)

# Build CNN model
model = models.Sequential([
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(train_data, epochs=10)

# Save model
model.save("models/medicine_model.h5")

print("Model training completed and saved!")