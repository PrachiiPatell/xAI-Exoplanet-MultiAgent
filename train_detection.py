import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load processed data
fluxes = np.load('processed_fluxes.npy')
labels = np.load('labels.npy')

# Reshape for CNN
fluxes = fluxes[..., np.newaxis]  # Shape: (24, 2000, 1)

# Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(fluxes, labels, test_size=0.2, random_state=42)

# Model
model = models.Sequential([
    layers.Input(shape=(2000, 1)),
    layers.Conv1D(8, 3, activation='relu'),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=20, batch_size=4, validation_data=(X_test, y_test))
model.save('planet_detector.keras')

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.legend()
plt.title('CNN Training for Exoplanet Detection')
plt.savefig('detection_accuracy.png')
plt.show()