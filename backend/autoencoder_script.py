import sys
import io
import json
import numpy as np
from keras.models import load_model
from keras.models import Model
import cv2
from pathlib import Path
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load your pre-trained model
autoencoder = load_model('model.h5')

# Load the training images
x_train = []
train_path = Path('./Train')
for t in range(925):
    image_path = train_path / f'pokemon ({t+1}).png'
    image = cv2.imread(str(image_path))
    if image is not None:
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x_train.append(img_rgb)

x_train = np.array(x_train)
x_train = x_train.astype('float32') / 255

# Get the encoder part of the autoencoder
encoder = Model(inputs=autoencoder.input,
                outputs=autoencoder.get_layer('encoder').output)

# Pre-encode all training images
encoded_imgs = encoder.predict(x_train, verbose=0)


def analyze_image(file_path):
    # Load and preprocess the input image
    image = cv2.imread(file_path)
    if image is None:
        return {"error": "Failed to load image"}
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x_test = np.array([img_rgb])
    x_test = x_test.astype('float32') / 255

    # Encode the input image
    encoded_img_test = encoder.predict(x_test, verbose=0)

    # Compare with all training images
    result = np.zeros((1, len(x_train)))
    for tr in range(len(x_train)):
        result[0][tr] = np.sum(abs(encoded_img_test[0] - encoded_imgs[tr]))

    # Find the most similar image
    most_similar_index = np.argmin(result[0])
    confidence = 1 - (result[0][most_similar_index] / np.max(result[0]))
    most_similar_image_path = str(
        train_path / f'pokemon ({most_similar_index + 1}).png')

    return {
        "confidence": float(confidence),
        "pokemon": f"Pokemon_{most_similar_index + 1}",
        "similar_image_index": int(most_similar_index),
        "similar_image_path": most_similar_image_path
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = analyze_image(file_path)
        print(json.dumps(result))
    else:
        print(json.dumps({"error": "No file path provided"}))
