import tensorflow as tf
import numpy as np
import cv2
import os

def predict_upload(model, class_names):
    print("Masukkan NAMA file gambar di folder project:")
    filename = input("Nama gambar (contoh: daun1.png): ")

    # Path otomatis ke folder project (tempat file prediksi.py berada)
    img_path = os.path.join(os.getcwd(), filename)

    # Load gambar
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Gambar tidak ditemukan! Pastikan nama file benar.")
        print(f"Dicari di: {img_path}")
        return
    
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_array = img_rgb.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    pred = model.predict(img_array)
    class_id = np.argmax(pred)
    confidence = np.max(pred)

    print("\n=== HASIL PREDIKSI ===")
    print(f"Prediksi     : {class_names[class_id]}")
    print(f"Kepercayaan  : {confidence*100:.2f}%")
    print(f"File dicek   : {img_path}")
