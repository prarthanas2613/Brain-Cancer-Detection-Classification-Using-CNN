import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# Define paths
dataset_path = r"C:\Users\HP\Documents\BrainCancerDetection\dataset\classification"
train_dir = os.path.join(dataset_path, "Training")
test_dir = os.path.join(dataset_path, "Testing")
output_path = r"C:\Users\HP\Documents\BrainCancerDetection\dataset\preprocessed_classification"
os.makedirs(output_path, exist_ok=True)

categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
num_classes = len(categories)  # Number of classes

# Function to load images and labels
def load_images_and_labels(data_dir, categories, target_size=(240, 240)):
    image_data = []
    labels = []

    for category_index, category in enumerate(categories):
        folder_path = os.path.join(data_dir, category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: {img_path} could not be read.")
                continue

            # Preprocessing
            img = cv2.resize(img, target_size)
            img = img.astype(np.float32) / 255.0  # Normalize
            img = cv2.medianBlur(img, 5)  # Apply blur if needed

            # Apply thresholding
            _, thresh_img = cv2.threshold((img * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

            # Apply morphological dilation (preserve float format)
            kernel = np.ones((5, 5), np.uint8)
            morph_img = cv2.dilate(thresh_img, kernel, iterations=1).astype(np.float32) / 255.0

            image_data.append(morph_img)
            labels.append(category_index)  # Assign integer label

            # Save preprocessed image
            output_img_path = os.path.join(output_path, f"{category}_{img_name}")
            cv2.imwrite(output_img_path, (morph_img * 255).astype(np.uint8))

    return np.array(image_data), np.array(labels)

# Load datasets
X_train, y_train = load_images_and_labels(train_dir, categories)
X_test, y_test = load_images_and_labels(test_dir, categories)

# Expand dimensions for CNN
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# One-hot encode labels
y_train_encoded = to_categorical(y_train, num_classes=num_classes)
y_test_encoded = to_categorical(y_test, num_classes=num_classes)

# Print dataset shapes
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train_encoded.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Testing labels shape: {y_test_encoded.shape}")

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(240, 240, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks to prevent overfitting
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint("best_brain_tumor_model.keras", save_best_only=True, monitor='val_accuracy', mode='max')
]

# Train model
history = model.fit(X_train, y_train_encoded, epochs=1, batch_size=32, 
                    validation_data=(X_test, y_test_encoded), callbacks=callbacks)

# Plot accuracy and loss curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over epochs')

plt.show()

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Generate classification report and confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_true = np.argmax(y_test_encoded, axis=1)

print("Classification Report:\n", classification_report(y_true, y_pred_classes, target_names=categories))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Save final model
model.save("brain_tumor_classification_model.keras")
model.save("brain_tumor_classification_model.h5")  # Save as .h5 for compatibility
