import tensorflow as tf

# Load model
model = tf.keras.models.load_model("models/medicine_model.h5")

IMG_SIZE = 128
BATCH_SIZE = 8

# Load test dataset
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Evaluate model
loss, accuracy = model.evaluate(test_data)

print("Test Accuracy:", accuracy)