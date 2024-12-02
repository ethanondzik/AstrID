import sys
import os
import numpy as np
import cv2
import datetime
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerPatch
from tensorflow.keras import backend as K
import tensorflow as tf
from keras.initializers import he_uniform

# Import custom model function
from models.unet import unet_model

# Get demo image to test the model
def extractImageFromFits(fits_file):
    with fits.open(fits_file) as hdul:
        image_data = hdul[0].data
    return image_data

def stackImages(images):
    stacked_images = np.array([np.stack([image, image, image], axis=-1) for image in images])

    prepared_images = np.array(stacked_images)

    return prepared_images

def extrackStarPredictions(prediction, threshold=0.4):
    # Normalize the prediction array to be between 0 and 1
    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())

    # Ensure the prediction array is 2D
    if prediction.ndim == 3:
        prediction = prediction[:, :, 0]

    # Threshold the prediction array to get the star locations
    stars = np.argwhere(prediction > threshold)

    # Create a list to store the star data
    star_data = []

    # Create a prediction mask of the same shape as the prediction array
    prediction_mask = np.zeros_like(prediction, dtype=np.uint8)

    # Iterate over the star locations and add them to the star data list and prediction mask
    for star in stars:
        y, x = star
        star_data.append((x, y))
        prediction_mask[y, x] = 1

    return star_data, prediction_mask


# Plot the subplot results from the model
def showPredictionComparison(image, test_image, model, threshold=0.4, save_prediction=False):
    pred_mask = model.predict(np.expand_dims(test_image, axis=0))[0]

    # Normalize the prediction array to be between 0 and 1
    pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())

    # Apply the threshold to create a binary mask
    pred_mask = (pred_mask > threshold).astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    ax[0].imshow(image, cmap='gray', origin='lower')
    ax[0].set_title('Image')

    ax[1].imshow(pred_mask, cmap='gray', origin='lower')
    ax[1].set_title('Prediction')

    plt.axis('off')
    plt.show()

    file_path = 'predictions/'
    filename = 'prediction_comparison.png'
    if save_prediction:
        # Date and time
        now = datetime.datetime.now()
        date_time = now.strftime("%Y_%m_%d-%H%M%S_")
        plt.savefig(file_path + date_time + filename)


def showPredictionOverlay(image, test_image, model, threshold=0.4, save_prediction=False):
    pred_star_data, prediction_mask = extrackStarPredictions(model.predict(np.expand_dims(test_image, axis=0))[0], threshold=threshold)
    print(np.count_nonzero(prediction_mask))


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # Plot the image
    ax.imshow(image, cmap='gray', origin='lower')

    # Draw red circles on the image for star predictions
    x_dim, y_dim = image.shape

    # Pixel-mask of stars
    pixel_mask = np.zeros((x_dim, y_dim))

    print('Drawing')  # DEBUG

    for star in pred_star_data:
        pixel_coords = star
        # Ensure the pixel coordinates are within bounds
        x, y = int(np.round(pixel_coords[0])), int(np.round(pixel_coords[1]))
        if 0 <= x < x_dim and 0 <= y < y_dim:
            pixel_mask[x][y] = 1

        Drawing_colored_circle = plt.Circle((pixel_coords[0], pixel_coords[1]), 4, fill=False, edgecolor='red', linewidth=0.2)
        ax.add_artist(Drawing_colored_circle)

    ax.set_title(f'{"Image with Star Prediction Overlay"}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(color='white', ls='dotted')

    # Add legend
    def make_legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        return Circle((width / 2, height / 2), 0.25 * height, fill=False, edgecolor=orig_handle.get_edgecolor(), linewidth=orig_handle.get_linewidth())


    # Display a legend for the circles
    blue_circle = Circle((0, 0), 1, fill=False, edgecolor='blue', linewidth=1)
    red_circle = Circle((0, 0), 1, fill=False, edgecolor='red', linewidth=1)
    ax.legend([blue_circle, red_circle], ['Star Location', 'Star Prediction'], loc='upper right', handler_map={Circle: HandlerPatch(patch_func=make_legend_circle)})
    
    plt.axis('off')
    plt.show()

    file_path = 'predictions/'
    filename = 'predictions_overlay.png'
    if save_prediction:
        # Date and time
        now = datetime.datetime.now()
        date_time = now.strftime("%Y_%m_%d-%H%M%S_")
        plt.savefig(file_path + date_time + filename)






def main(image_path):
    # Check if the image file exists
    image_path = image_path
    print("Image path: ", image_path)
    
    model_weights = []
    
    # Load the trained model weights
    model_weights = 'models/model_weights/FINAL_2024_11_29-0023_24_unet_model_chris_model_weights.h5'
    print("Model weights: ", model_weights)


    # Define hyperparameters
    hyperparameters = {
        'input_shape': (512, 512, 3),
        'filters': [64, 128, 256, 512, 1024],
        'kernel_size': (3, 3),
        'activation': 'relu',
        'padding': 'same',
        'initializer': he_uniform
    }

    # Create and compile the model using hyperparameters
    model = unet_model(
        input_shape=hyperparameters['input_shape'],
        filters=hyperparameters['filters'],
        kernel_size=hyperparameters['kernel_size'],
        activation=hyperparameters['activation'],
        padding=hyperparameters['padding'],
        initializer=hyperparameters['initializer']
    )

    # Load the saved model weights
    model.load_weights(model_weights)


    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Resize the image to 512x512 if necessary
    if image.shape != (512, 512):
        image = cv2.resize(image, (512, 512))

    # Convert the image to a numpy array
    np.expand_dims(image, axis=0)

    # Normalize the image
    image = image / 255.0


    # Check for GPU availability
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("GPU is available")
        K.clear_session()
        tf.config.experimental.reset_memory_stats('GPU:0')


    # Show the prediction comparison
    showPredictionComparison(image, image, model, threshold=0.4, save_prediction=True)

    # Show the prediction overlay
    showPredictionOverlay(image, image, model, threshold=0.4, save_prediction=True)
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python demoModel.py <path_to_image>")
    else:
        main(sys.argv[1])