import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import matplotlib.pyplot as plt

# Define paths
csv_path = r'C:\Users\Duaa.Aqel\PycharmProjects\PredictPringles_with_color\venv\Pringles-Object-Detection-2\train\_annotations.csv'
image_dir = r'C:\Users\Duaa.Aqel\PycharmProjects\PredictPringles_with_color\venv\Pringles-Object-Detection-2\train'
model_save_path = r'C:\Users\Duaa.Aqel\PycharmProjects\PredictPringles_with_color\saved_model'

# Function definitions remain the same as in your previous script

def load_dataset(csv_path, image_dir):
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {csv_path}")
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found. Please check the path and try again.")
        return None, None, None

    class_mapping = {label: idx for idx, label in enumerate(df['class'].unique())}
    df['class'] = df['class'].map(class_mapping)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, val_df, class_mapping

def load_image(path, bbox, target_size=(128, 128)):
    try:
        img = load_img(path)
        img = img_to_array(img)
        xmin, ymin, xmax, ymax = bbox
        img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        img = tf.image.resize(img, target_size)
        return img / 255.0
    except FileNotFoundError:
        print(f"Error: The image file {path} was not found.")
        return None

def create_dataset(df, image_dir):
    images = []
    labels = []
    for _, row in df.iterrows():
        path = os.path.join(image_dir, row['filename'])
        bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        img = load_image(path, bbox)
        if img is not None:
            images.append(img)
            labels.append([row['class']])
    return np.array(images), np.array(labels)

def build_model(input_shape, num_classes):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    class_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='class_output')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=class_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_images, train_labels, val_images, val_labels, epochs=10):
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels))
    model.save(model_save_path)
    return history

def evaluate_model(model, val_images, val_labels):
    loss, accuracy = model.evaluate(val_images, val_labels)
    print(f'Validation Loss: {loss}')
    print(f'Validation Accuracy: {accuracy}')

def plot_image_with_bbox(img, label):
    plt.imshow(img)
    plt.title(f'Predicted: {label}')
    plt.show()

# Load and preprocess dataset
train_df, val_df, class_mapping = load_dataset(csv_path, image_dir)
if train_df is None or val_df is None:
    exit()

# Create datasets
train_images, train_labels = create_dataset(train_df, image_dir)
val_images, val_labels = create_dataset(val_df, image_dir)

# Ensure there are no missing images
if len(train_images) == 0 or len(val_images) == 0:
    print("Error: No images found. Please check the paths and try again.")
    exit()

# Build model
input_shape = train_images[0].shape
num_classes = len(class_mapping)
model = build_model(input_shape, num_classes)
model.summary()

# Train model
history = train_model(model, train_images, train_labels, val_images, val_labels)

# Evaluate model
evaluate_model(model, val_images, val_labels)
