import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------- Load and Preprocess Dataset ----------------------
dataset_path_detection = r"C:\Users\HP\Documents\New folder\BrainCancerDetection\dataset\detection"
categories_detection = ['yes', 'no']  # 'yes' for tumor, 'no' for non-tumor
image_data, labels = [], []

# Load images and labels
for category in categories_detection:
    folder_path = os.path.join(dataset_path_detection, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        
        if img is None:
            print(f"Warning: {img_path} could not be read.")
            continue
        
        # Preprocessing
        img = cv2.resize(img, (240, 240))  # Standard shape
        img = (img / 255.0).astype(np.float32)  # Normalize
        
        image_data.append(img)
        labels.append(1 if category == 'yes' else 0)

# Convert to NumPy arrays
image_data = np.array(image_data)
labels = np.array(labels)

# ---------------------- Train-Test Split ----------------------
X_train, X_temp, y_train, y_temp = train_test_split(image_data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Expand dimensions for CNN (needed for grayscale images)
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# ---------------------- Improved Data Augmentation ----------------------
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],  # Adjust brightness
    channel_shift_range=0.2,  # Simulates contrast changes
    horizontal_flip=True
)
datagen.fit(X_train)

# ---------------------- Deeper & Stronger CNN Model ----------------------
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(240, 240, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),  # Lower LR for better learning
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ---------------------- Learning Rate Scheduling & Early Stopping ----------------------
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ---------------------- Model Training ----------------------
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=1,  # Increased epochs
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler, early_stopping]
)

# ---------------------- Plot Accuracy & Loss ----------------------
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

# ---------------------- Model Evaluation ----------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")  # Convert probabilities to class labels

print(classification_report(y_test, y_pred_classes, target_names=['no', 'yes']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ---------------------- Save Model ----------------------
model.save("brain_tumor_detection_model.keras")
