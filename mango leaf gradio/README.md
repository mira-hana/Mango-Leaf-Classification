# 🍃 Mango Leaf Disease Classifier

A web app built using Gradio to classify mango leaf diseases.  
It supports both users and researchers.

## Features

- Upload an image to get disease prediction.
- Researchers can upload labeled images for retraining.
- Includes script to retrain the model using new data.

## How to Run in Google Colab -> Use this code below to run it

#1. Clone the GitHub repository
!git clone https://github.com/mira-hana/Mango-Leaf-Classification.git

#2. Change directory to the Gradio app folder
%cd "Mango-Leaf-Classification/mango leaf gradio"

#3. Install required packages
!pip install -r requirements.txt

#4. Launch the Gradio app
!python app.py
