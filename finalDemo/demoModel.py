import sys
import os
import numpy as np
import datetime
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerPatch
from keras.initializers import he_uniform

# Import custom model function
from models.unet import unet_model

# Get demo image to test the model
def extractImageFromFits(fits_file):
    with fits.open(fits_file) as hdul:
        image_data = hdul[0].data
    return image_data

# Get pixel mask from fits file
def extractPixelMaskFromFits(fits_file):
    with fits.open(fits_file) as hdul:
        pixel_mask = hdul['pixel_mask'].data
        return pixel_mask

def stackImages(images):
    stacked_images = np.array([np.stack([image, image, image], axis=-1) for image in images])

    prepared_images = np.array(stacked_images)

    return prepared_images

def extractStarPredictions(prediction, threshold=0.4):
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

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].imshow(image, cmap='gray', origin='lower')
    ax[0].set_title('Image')

    ax[1].imshow(pred_mask, cmap='gray', origin='lower')
    ax[1].set_title('Prediction')

    plt.axis('off')

    file_path = 'predictions/'
    filename = 'prediction_comparison.png'
    if save_prediction:
        # Date and time
        now = datetime.datetime.now()
        date_time = now.strftime("%Y_%m_%d-%H%M%S_")
        plt.savefig(file_path + date_time + filename)

    plt.show()




def showPredictionOverlay(image, test_image, model, threshold=0.4, save_prediction=False):
    pred_star_data, prediction_mask = extractStarPredictions(model.predict(np.expand_dims(test_image, axis=0))[0], threshold=threshold)
    print("Number of stars detected:", len(pred_star_data))


    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    # Plot the image
    ax.imshow(image, cmap='gray', origin='lower')

    # Draw red circles on the image for star predictions
    x_dim, y_dim = image.shape

    # Pixel-mask of stars
    pixel_mask = np.zeros((x_dim, y_dim))

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

    file_path = 'predictions/'
    filename = 'predictions_overlay.png'
    if save_prediction:
        # Date and time
        now = datetime.datetime.now()
        date_time = now.strftime("%Y_%m_%d-%H%M%S_")
        plt.savefig(file_path + date_time + filename)

    plt.show()


def main():

    # Load the image for testing    
    test_image = np.load('test_image.npy')
    print("Test image shape: ", test_image.shape)
    
    # Select the image to display
    display_image = test_image[:, :, 0]
    print("Display image shape: ", display_image.shape)
    
    # Load the trained model weights
    model_weights = "FINAL_2024_11_29-0023_24_unet_model_chris_model_weights.h5"


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

    # Show the prediction comparison
    showPredictionComparison(display_image, test_image, model, threshold=0.5, save_prediction=True)

    # Show the prediction overlay
    showPredictionOverlay(display_image, test_image, model, threshold=0.5, save_prediction=True)
    

if __name__ == "__main__":
    main()