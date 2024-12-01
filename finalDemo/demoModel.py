import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf

from scripts.imageProcessing import stackImages

def main(image_path):
    # Check if the image file exists
    if not os.path.isfile(image_path):
        print(f"Error: The file {image_path} does not exist.")
        return

    # Load the trained model
    saved_models_path = 'models/saved_models/'
    model_files = [file for file in os.listdir(saved_models_path) if file.endswith('.keras')]
    model_files = sorted(model_files)
    model_choice = model_files[-1]
    model = load_model(os.path.join(saved_models_path, model_choice))

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: The file {image_path} is not a valid image.")
        return

    # Expand dimensions to match the model input
    image_input = stackImages(image)

    # Check for GPU availability
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("GPU is available")
        K.clear_session()
        tf.config.experimental.reset_memory_stats('GPU:0')

    # Make prediction
    prediction = model.predict(image_input)

    # Threshold the prediction to create a binary mask
    threshold = 0.4
    prediction_mask = (prediction > threshold).astype(np.uint8)

    # Find contours of the prediction mask
    contours, _ = cv2.findContours(prediction_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red circles around the predicted areas on the original image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(output_image, center, radius, (0, 0, 255), 2)

    # Save the output image
    output_path = os.path.splitext(image_path)[0] + '_prediction.png'
    cv2.imwrite(output_path, output_image)
    print(f"Prediction saved to {output_path}")

    # Display the input image and the output image with predictions
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_image)
    plt.title('Prediction Overlay')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python demoModel.py <path_to_image>")
    else:
        main(sys.argv[1])