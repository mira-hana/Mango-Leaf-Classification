# retrain.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os

DATASET_DIR = "dataset"
MODEL_PATH = "model/mobilenetv2_mango_leaf.h5"
UPDATED_MODEL_PATH = "model/mobilenetv2_updated.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                        class_mode="categorical", subset="training")
val_gen = datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                      class_mode="categorical", subset="validation")

model = load_model(MODEL_PATH)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

model.save(UPDATED_MODEL_PATH)
print(f"Updated model saved to {UPDATED_MODEL_PATH}")