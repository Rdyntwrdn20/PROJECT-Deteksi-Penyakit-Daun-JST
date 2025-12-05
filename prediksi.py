import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2

IMG_SIZE = 224

def predict_upload(model, class_names):
    print("\n=== PREDIKSI GAMBAR ===")
    path = input("Masukkan path gambar: ")

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # PREPROCESS MOBILENET
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    idx = np.argmax(pred)
    
    print("\nHasil Prediksi:", class_names[idx])
    print("Confidence:", round(np.max(pred)*100, 2), "%")

    return class_names[idx]
