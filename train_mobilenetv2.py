# train_mobilenetv2.py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----- KONFIGURASI -----
DATA_DIR = "DATASET"
IMG_SIZE = (224, 224)  # MobileNetV2 rekomendasi 160/192/224 -> pakai 224
BATCH_SIZE = 32
SEED = 123
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE
MODEL_SAVE_PATH = "mobilenetv2_tomato.h5"

# ----- 1) Muat dataset (train/validation split) -----
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="training"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="validation"
)

class_names = train_ds.class_names
print("Class names:", class_names)

# ----- 2) Perhitungan class weights (jika imbalance) -----
# Hitung jumlah gambar tiap kelas dari train_ds (sederhana)
counts = {}
for cls in class_names:
    counts[cls] = 0

for images, labels in train_ds.unbatch().map(lambda x,y: (x,y)).batch(10000):
    for lbl in labels.numpy():
        counts[class_names[lbl]] += 1

print("Counts per class (train):", counts)
y_train_all = np.concatenate([y.numpy() for x,y in train_ds.unbatch().batch(10000)])
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_all), y=y_train_all)
class_weights = {i: cw[i] for i in range(len(cw))}
print("Class weights:", class_weights)

# ----- 3) Preprocessing & Augmentasi -----
# Normalisasi sesuai MobileNetV2 (preprocess_input) atau rescale to [0,1]
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess(image, label):
    image = preprocess_input(image)  # -> [-1,1] sesuai MobileNetV2
    return image, label

# Data augmentation sebagai model layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.05, 0.05),
])

# Terapkan preprocess & cache/prefetch
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y), num_parallel_calls=AUTOTUNE)

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

# Tambah augmentasi hanya pada training
train_ds = train_ds.map(lambda x,y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ----- 4) Bangun model MobileNetV2 (transfer learning) -----
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # freeze initial

# Head classifier
inputs = layers.Input(shape=IMG_SIZE + (3,))
x = inputs
# optional: augmentation already applied above, but keep identity here
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)

model.summary()

# ----- 5) Compile & callbacks -----
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint("best_"+MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

# ----- 6) TRAINING (fase 1: head only) -----
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    class_weight=class_weights
)

# ----- 7) Fine-tune (opsional): buka beberapa layer terakhir dari base_model -----
# Fine-tuning recommended: unfreeze a small chunk
base_model.trainable = True
# tentukan berapa banyak layer dari belakang yg ingin di-train
fine_tune_at = int(len(base_model.layers) * 0.7)  # unfreeze top 30%
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 10
total_epochs = EPOCHS + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] if hasattr(history, 'epoch') else 0,
    validation_data=val_ds,
    callbacks=callbacks,
    class_weight=class_weights
)

# ----- 8) Simpan model akhir -----
model.save(MODEL_SAVE_PATH)
print("Model saved to", MODEL_SAVE_PATH)

# ----- 9) Visualisasi training -----
def plot_history(his, start_epoch=0):
    acc = his.history.get('accuracy', [])
    val_acc = his.history.get('val_accuracy', [])
    loss = his.history.get('loss', [])
    val_loss = his.history.get('val_loss', [])
    epochs_range = range(start_epoch, start_epoch + len(acc))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='train_acc')
    plt.plot(epochs_range, val_acc, label='val_acc')
    plt.legend()
    plt.title('Accuracy')
    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label='train_loss')
    plt.plot(epochs_range, val_loss, label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

plot_history(history)
plot_history(history_fine, start_epoch=len(history.history.get('loss', [])))
