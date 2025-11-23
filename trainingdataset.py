import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# --- KONFIGURASI ---
DATA_DIR = 'dataset'   # Nama folder tempat gambar
IMG_SIZE = 128         # Ukuran gambar input (128x128 pixel)
BATCH_SIZE = 32        # Jumlah gambar yang diproses dalam satu batch

print("=== MULAI MEMUAT DATASET ===")

# 1. Memuat Data Training (80% dari total gambar)
# Ini adalah data yang akan digunakan model untuk 'belajar'
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,   # Ambil 20% untuk validasi, sisanya training
    subset="training",
    seed=123,               # Seed agar acakan konsisten
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# 2. Memuat Data Validasi (20% sisanya)
# Ini adalah data untuk 'ujian' model agar tidak menghafal
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# 3. Menyimpan Nama Kelas
class_names = train_ds.class_names
print(f"\nKelas yang ditemukan: {class_names}")
print(f"Jumlah kelas: {len(class_names)}")

# Optimasi performa (caching) agar training lebih cepat
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("=== DATASET SIAP DIGUNAKAN ===")

# ... (Pastikan kode Bagian 1 sebelumnya tetap ada di atas) ...

print("\n=== MEMBANGUN ARSITEKTUR CNN ===")

# 4. Definisi Model Sequential (Tumpukan Layer)
model = models.Sequential([
    # Layer Pre-processing: Rescaling
    # Mengubah nilai piksel gambar dari 0-255 menjadi 0-1 agar ringan dihitung
    layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Layer 1: Konvolusi & Pooling
    layers.Conv2D(32, (3, 3), activation='relu'), # 32 Filter
    layers.MaxPooling2D((2, 2)),

    # Layer 2: Konvolusi & Pooling
    layers.Conv2D(64, (3, 3), activation='relu'), # 64 Filter
    layers.MaxPooling2D((2, 2)),

    # Layer 3: Konvolusi & Pooling
    layers.Conv2D(128, (3, 3), activation='relu'), # 128 Filter
    layers.MaxPooling2D((2, 2)),

    # Layer Klasifikasi (Fully Connected)
    layers.Flatten(),                       # Ubah matriks jadi satu baris lurus
    layers.Dense(128, activation='relu'),   # Hidden layer 'berpikir'
    layers.Dropout(0.5),                    # Mencegah menghafal (Overfitting)
    layers.Dense(len(class_names), activation='softmax') # Output layer (3 Pilihan)
])

# Menampilkan ringkasan model di terminal
model.summary()

# 5. Compile Model (Menentukan cara belajar)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

print("\n=== MULAI TRAINING (LATIHAN) ===")
# 6. Proses Training
# Epochs = 15 artinya model akan melihat seluruh materi dataset sebanyak 15 kali ulang
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

# 7. Simpan Model
# Ini penting! Biar nanti pas demo kita ga perlu training ulang
model.save('model_tomat.h5')
print("\n=== TRAINING SELESAI & MODEL DISIMPAN SBG 'model_tomat.h5' ===")