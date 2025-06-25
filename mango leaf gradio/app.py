import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import uuid

# Load model
model = tf.keras.models.load_model("mobilenetv2_mango_leaf.h5")

# Set class names
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# Prediction function for 'User'
def predict(img: Image.Image):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    return f"Predicted: {predicted_class} ({confidence:.2f}%)"

# Save function for 'Researcher'
def save_labeled_data(img: Image.Image, label: str):
    save_dir = os.path.join("dataset", label)
    os.makedirs(save_dir, exist_ok=True)

    # Save image with a unique filename
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(save_dir, filename)
    img.save(path)

    return f"✅ Image saved under class '{label}' as '{filename}'"

# Gradio UI setup
def role_selector(role):
    if role == "User":
        return gr.Interface(
            fn=predict,
            inputs=gr.Image(type="pil"),
            outputs=gr.Textbox(),
            title="🍃 Mango Leaf Disease Classifier",
            description="Upload a mango leaf image to detect disease using MobileNetV2 model.",
        )

    elif role == "Researcher":
        return gr.Interface(
            fn=save_labeled_data,
            inputs=[
                gr.Image(type="pil", label="Upload Mango Leaf Image"),
                gr.Dropdown(class_names, label="Select Correct Label")
            ],
            outputs=gr.Textbox(),
            title="📥 Research Data Submission",
            description="Contribute labeled images to improve the model.",
        )

# Main role selector interface
demo = gr.TabbedInterface(
    interface_list=[
        role_selector("User"),
        role_selector("Researcher")
    ],
    tab_names=["User", "Researcher"]
)

# Launch in Colab or local
demo.launch(share=True)
