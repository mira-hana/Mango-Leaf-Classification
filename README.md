**🥭 Mango Leaf Disease Detection**

> *Computer Vision | Deep Learning | Transfer Learning*

MangoVis is a deep learning project that classifies *mango leaf diseases* from images using *Simple CNN, MobileNetV2, and EfficientNetB0*.

*🌱 Dataset*

- 4,000+ labeled images
- 8 classes: 7 diseases + Healthy
- Images resized to **128×128**
- Normalization and augmentation applied
- 70% training / 30% testing

*🧠 Models*

| Model              | Purpose                           |
|--------------------|-----------------------------------|
| **Simple CNN**     | Baseline model                    |
| **MobileNetV2**    | Lightweight and fast              |
| **EfficientNetB0** | Transfer learning and fine-tuning |

Models were evaluated using *Accuracy, Precision, Recall, and F1-Score*.

*🖥️ Deployment*

The selected model is deployed with *Gradio*, providing:

1) *User interface:* Upload an image and receive a disease prediction.
<img width="1283" height="687" alt="image" src="https://github.com/user-attachments/assets/9e5ba3b9-29e2-4117-89d8-378943915fed" />

2) *Researcher interface:* Manually label images for potential future model improvements.
<img width="1280" height="692" alt="image" src="https://github.com/user-attachments/assets/65523555-c029-45c1-ab70-32787459b383" />

*🛠️ Technologies*

Python · TensorFlow/Keras · CNN · MobileNetV2 · EfficientNetB0 · OpenCV/PIL · Scikit-learn · Gradio

*🔄 Workflow*

Dataset 

↓ 

Preprocessing & Augmentation 

↓ 

CNN / MobileNetV2 / EfficientNetB0 

↓ 

Model Evaluation & Comparison 

↓ 

Selected Model 

↓ 

Gradio Deployment

*💡 Skills*

Computer Vision · Deep Learning · Transfer Learning · Fine-Tuning · Image Processing · Model Evaluation · AI Deployment
