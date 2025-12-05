import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 224
TEST_DIR = "DATASET/test"

# ==========================
# LOAD MODEL
# ==========================
model = tf.keras.models.load_model("mobilenetv2_best.keras")

# ==========================
# LOAD TEST SET
# ==========================
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    shuffle=False
)

class_names = test_ds.class_names
print("Class Names:", class_names)

# PREPROCESS MOBILE NET
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
test_ds = test_ds.map(lambda x, y: (preprocess(x), y))

# ==========================
# PREDIKSI
# ==========================
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

# ==========================
# CONFUSION MATRIX
# ==========================
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================
# CLASSIFICATION REPORT
# ==========================
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
