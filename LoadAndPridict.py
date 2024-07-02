import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import matplotlib.pyplot as plt

# Define paths
model_load_path = r'C:\Users\Duaa.Aqel\PycharmProjects\PredictPringles_with_color\saved_model'

def load_image_for_prediction(image_path, target_size=(128, 128)):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except FileNotFoundError:
        print(f"Error: The image file {image_path} was not found.")
        return None

def predict_model_single_image(model, image_array, class_mapping):
    prediction = model.predict(image_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_label = list(class_mapping.keys())[predicted_class_idx]
    color = ' '
    if  predicted_label == 'pringles_onion':
        color = 'Green_pringles'
    elif  predicted_label == 'pringles_original':
        color = 'Red_pringles'
    elif  predicted_label == 'pringles_churrasco':
        color = 'Brown_pringles'
    return color

def plot_image_with_prediction(image_path, label):
    img = load_img(image_path)
    plt.imshow(img)
    plt.title(f'Predicted: {label}')
    plt.show()

# Load the saved model
model = tf.keras.models.load_model(model_load_path)

# Define class mapping manually if not available in saved file
class_mapping = {'pringles_onion': 0, 'pringles_original': 1, 'pringles_churrasco': 2}  # Adjust according to your dataset

# Predict on a single image
image_path = r'C:\Users\Duaa.Aqel\PycharmProjects\PredictPringles_with_color\venv\Pringles-Object-Detection-2\test\00000093_jpg.rf.d264fdf893c2c60e248ce03384455d2b.jpg'  # Replace with your image file path
image_array = load_image_for_prediction(image_path)
if image_array is not None:
    predicted_label = predict_model_single_image(model, image_array, class_mapping)
    plot_image_with_prediction(image_path, predicted_label)
