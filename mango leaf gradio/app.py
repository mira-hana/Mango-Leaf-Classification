import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Set up page
st.set_page_config(page_title="Mango Leaf Disease Classifier", layout="centered")
st.title("🍃 Mango Leaf Disease Classifier (MobileNetV2)")
st.write("Upload a mango leaf image to detect the disease using MobileNetV2.")

# Load the MobileNetV2 model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenetv2_mango_leaf.h5")

model = load_model()

# Class labels (update to match your model's output)
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Powdery Mildew', 'Sooty Mould', 'Healthy']

# Upload image
uploaded_file = st.file_uploader("Upload Mango Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess image
    def preprocess(img, target_size=(224, 224)):
        img = img.resize(target_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    img_array = preprocess(img)

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Show result
    st.subheader("🔍 Prediction Result")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
