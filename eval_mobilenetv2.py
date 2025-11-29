import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# --- KONFIGURASI ---
MODEL_PATH = "best_mobilenetv2_tomato.h5"
DATA_DIR = "dataset"
IMG_SIZE = 224
BATCH_SIZE = 32

# --- LOAD MODEL ---
if not os.path.exists(MODEL_PATH):
    print("ERROR: Model tidak ditemukan!")
    exit()

print("Memuat model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# --- LOAD DATASET VALIDASI SAJA ---
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = val_ds.class_names
print("Kelas:", class_names)

# --- AMBIL LABEL ASLI ---
y_true = []
for _, labels in val_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

# --- PREDIKSI ---
y_pred = model.predict(val_ds)
y_pred_classes = np.argmax(y_pred, axis=1)

# --- CONFUSION MATRIX ---
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix (Model MobileNetV2 yang Sudah Diperbaiki)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- CLASSIFICATION REPORT ---
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# --- VISUALISASI CONTOH PREDIKSI ---
import random

plt.figure(figsize=(12,8))

for i in range(9):
    idx = random.randint(0, len(y_true)-1)
    img_batch, label_batch = next(iter(val_ds))

    img = img_batch[idx].numpy().astype("uint8")
    true_label = class_names[y_true[idx]]
    pred_label = class_names[y_pred_classes[idx]]

    plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis("off")

plt.suptitle("Sample Prediksi Model (MobileNetV2 Fine-tuned)", fontsize=16)
plt.show()
