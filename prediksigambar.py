from tensorflow.keras.models import load_model
from predict_upload import predict_upload

model = load_model("mobilenetv2_best.keras")

class_names = [
    "Tomato__Bacterial_spot",
    "Tomato_Healthy",
    "Tomato_Yellow_leaf_curl_Virus"
]

predict_upload(model, class_names)
