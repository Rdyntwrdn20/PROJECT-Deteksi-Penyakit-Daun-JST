import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# ==========================
# KONFIGURASI
# ==========================
DATA_DIR = "DATASET"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

# ==========================
# LOAD DATASET
# ==========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = train_ds.class_names
print("Class Names:", class_names)


# ==========================
# PREFETCH
# ==========================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==========================
# PREPROCESSING
# ==========================
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input


train_ds = train_ds.map(lambda x, y: (preprocess(x), y)).prefetch(AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (preprocess(x), y)).prefetch(AUTOTUNE)
test_ds  = test_ds.map(lambda x, y: (preprocess(x), y)).prefetch(AUTOTUNE)

# ==========================
# ARSITEKTUR MODEL
# ==========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False   # freeze awal

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================
# CALLBACKS
# ==========================
checkpoint = ModelCheckpoint(
    "mobilenetv2_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-6
)

# ==========================
# TRAINING
# ==========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

model.save("mobilenetv2_final.h5")
print("Model final disimpan!")

# ==========================
# GRAFIK TRAINING
# ==========================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Akurasi Training")
plt.legend(["Train", "Val"])

plt.subplot(1,2,2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss Training")
plt.legend(["Train", "Val"])

plt.tight_layout()

# --- TAMBAHKAN BAGIAN INI ---
filename_grafik = "training_graph.png"
plt.savefig(filename_grafik) 
print(f"Grafik berhasil disimpan sebagai {filename_grafik}")
# ----------------------------

# ==========================
# CONFUSION MATRIX
# ==========================
print("\nMembuat Confusion Matrix...")

# Prediksi data test
y_pred_prob = model.predict(test_ds)
y_pred = np.argmax(y_pred_prob, axis=1)

# Label asli
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Hitung confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualisasi
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

plt.figure(figsize=(6,6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix MobileNetV2")
plt.tight_layout()

# ==========================
# TESTING EVALUATION
# ==========================
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Accuracy : {test_accuracy*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")


# Simpan gambar confusion matrix
plt.savefig("confusion_matrix.png")
print("Confusion matrix disimpan sebagai confusion_matrix.png")


plt.show()