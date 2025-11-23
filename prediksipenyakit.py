import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import tkinter as tk
from tkinter import filedialog
import os
import sys

# --- KONFIGURASI ---
MODEL_PATH = 'model_tomat.h5'

# 1. Memuat Model
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: File model '{MODEL_PATH}' tidak ditemukan!")
    sys.exit()

print("Sedang memuat model AI...")
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Urutan Kelas (SUDAH DIPERBAIKI SESUAI SCAN TENSORFLOW)
# Index 0: Bacterial
# Index 1: Virus (Karena 'T' besar menang lawan 'h' kecil)
# Index 2: healthy (Huruf 'h' kecil bikin dia jadi terakhir)
class_names = [
    'Tomato___Bacterial_spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___healthy' 
]

print(f"Kamus Kelas: {class_names}")

# 3. Pilih File
print("Silakan pilih file gambar...")
root = tk.Tk()
root.withdraw() 
file_path = filedialog.askopenfilename(
    title="Pilih Foto Daun Tomat",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if not file_path:
    print("Batal memilih file.")
    sys.exit()

# 4. Proses & Prediksi
try:
    print(f"\nMenganalisa: {os.path.basename(file_path)}")
    
    # Load & Resize
    img = load_img(file_path, target_size=(128, 128))
    img_array = img_to_array(img)
    
    # NORMALISASI MANUAL (PENTING!)
    # Berdasarkan eksperimen, modelmu bekerja lebih baik jika dibagi 255 manual
    img_array = img_array / 255.0 
    
    img_array = tf.expand_dims(img_array, 0) 

    # Prediksi
    predictions = model.predict(img_array)
    score = predictions[0]
    
    juara_index = np.argmax(score)
    juara_label = class_names[juara_index]
    persen = 100 * np.max(score)

    # Output
    print("\n" + "="*35)
    print(f"🔍 HASIL DIAGNOSA AI")
    print("="*35)
    print(f"Penyakit  : {juara_label}")
    print(f"Keyakinan : {persen:.2f}%")
    print("-" * 35)
    print("Detail Probabilitas:")
    for i, name in enumerate(class_names):
        print(f"- {name} : {score[i]*100:.2f}%")
    print("="*35 + "\n")

except Exception as e:
    print(f"Error: {e}")