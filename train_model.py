import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Set image size
IMG_SIZE = (150, 150)

# Image generators
datagen = ImageDataGenerator(rescale=1./255)

# Training data
train = datagen.flow_from_directory(
    'dataset/train',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='binary'
)

# Testing/validation data
val = datagen.flow_from_directory(
    'dataset/test',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='binary'
)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(train, validation_data=val, epochs=5)

# Save
model.save('model/muffin_or_chihuahua.h5')
print("✅ Model saved to model/muffin_or_chihuahua.h5")
