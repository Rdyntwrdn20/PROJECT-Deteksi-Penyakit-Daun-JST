import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# --- KONFIGURASI ---
DATA_DIR = 'dataset'
IMG_SIZE = 224   # MobileNetV2 wajib 160/192/224
BATCH_SIZE = 32

print("=== MEMUAT DATASET ===")

# Memuat dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Kelas:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ======================================================
# ===============  ARSITEKTUR MobileNetV2 ===============
# ======================================================

print("\n=== MEMBANGUN MODEL MobileNetV2 ===")

# 1. Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
])

# 2. Load MobileNetV2 (pretrained)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False   # freeze fitur bawaan mobilenet

# 3. Full model
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)  # fitur dari MobileNetV2

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.summary()

# --- TRAINING SETTING ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# EarlyStopping → hentikan jika sudah optimal
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Simpan model terbaik
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "model_tomat.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

print("\n=== MULAI TRAINING MODEL ===")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[early_stop, checkpoint]
)

print("\n=== TRAINING SELESAI ===")
print("Model terbaik disimpan sebagai 'model_tomat.h5'")
