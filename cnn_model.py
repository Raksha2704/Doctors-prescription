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

# SIMPLE augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
])

# SIMPLE CNN
model = models.Sequential([

    data_augmentation,

    layers.Rescaling(1./255),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),

    layers.Dense(len(class_names), activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(train_data, epochs=15)

# Save
model.save("models/medicine_model.h5")

print("Training completed!")