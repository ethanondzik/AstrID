# AstrID
*AstrID:* A project focused on identifying and classifying astronomical objects using data from various space catalogs. Leveraging machine learning, AstrID aims to enhance our understanding of stars, galaxies, and other celestial phenomena.

## Project Goals and Objectives
The primary goal of AstrID is to develop a robust system for identifying and classifying various astronomical objects, such as stars and galaxies, using data from space catalogs. A stretch goal of the project is to identify potential black hole candidates using advanced machine learning techniques.

## Features
- **Data Retrieval**: Fetch high-resolution images and data from space catalogs like Hipparcos and 2MASS.
- **Image Processing**: Process and visualize astronomical images with WCS overlays.
- **Machine Learning**: Classify different types of stars and other celestial objects using a U-Net model.
- **Model Training**: Train machine learning models using high-resolution astronomical images.
- **Model Evaluation**: Evaluate the performance of trained models on validation and test datasets.
- **Prediction**: Make predictions on new astronomical data using trained models.
- **Black Hole Identification**: (Stretch Goal) Identify potential black hole candidates.

## Instructions

### Installation
How to install the program and prepare for running.

**`NOTE`**` : This Repo will require Git LFS in order to download saved model weights files, as '.h5' files are over 100MB.`

Install Git LFS

    sudo apt-get install git-lfs

Initialize Git LFS

    git lfs install

Full installation instructions with GPU functionality can be found in the [`GPU_setup.md`](docs/GPU_setup.md)

1. **Navigate to the main folder and create a new virtual environment**:
    ```bash
    python3 -m venv .venv
    ```

2. **Activate the environment**:
    ```bash
    source .venv/bin/activate
    ```

3. **Install required packages**:
    ```bash
    pip install -r requirements.txt
    ```

### Viewing Notebooks

#### If viewing notebooks in VS Code:

1. **Install Python extension in VS Code**.
2. **Install Jupyter extension in VS Code**.
3. **Install Python kernel for your created venv**:
    ```bash
    python3 -m ipykernel install --user --name=.venv
    ```
4. **Select your installed kernel to be used with your notebook**: Click the "Select Kernel" button in the top right.

#### Alternatively:

Notebooks may be viewed in any application capable of handling *.ipynb files.

### System Dependencies

Ensure the following system dependencies are installed:
```bash
sudo apt-get install libgl1-mesa-glx
```

### Setting Up CUDA and cuDNN for TensorFlow GPU Support

**See instructions listed in [`GPU_setup.md`](docs/GPU_setup.md).**


## Usage

### Validating the Model

We use the `validateModel.ipynb` notebook to validate our trained U-Net model. The notebook includes the following steps:

1. **Importing Necessary Libraries and Modules**:
   - TensorFlow and Keras for loading and using the trained neural network.
   - NumPy for numerical operations and handling arrays.
   - Matplotlib for plotting and visualizing data.
   - Astropy for handling FITS files and WCS data.
   - Custom functions from `dataGathering` and `imageProcessing`.

2. **Loading the Dataset**:
   - Use the `importDataset` function to load the validation dataset and extract images, masks, star data, WCS data, and FITS file names.

3. **Preparing Images and Masks for the Model**:
   - Convert images to 3-channel format and masks to single-channel format.
   - Normalize the images using min-max normalization.

4. **Loading the Trained Model**:
   - Load the trained U-Net model from the saved models directory.

5. **Evaluating the Model**:
   - Evaluate the model's performance on the validation dataset by calculating loss and accuracy metrics.

6. **Making Predictions**:
   - Use the trained model to make predictions on the validation dataset.

7. **Visualizing Results**:
   - Visualize the results by plotting the original images, ground truth masks, and predicted masks.


### Training the Model

We use the `trainModel.ipynb` notebook to train our U-Net model. The notebook includes the following steps:

1. **Importing Necessary Libraries and Modules**:
   - TensorFlow and Keras for building and training the neural network.
   - NumPy for numerical operations and handling arrays.
   - Matplotlib for plotting and visualizing data.
   - Custom functions from `unet`, `dataGathering`, `imageProcessing`, and `log`.

2. **Initializing Lists for the Dataset**:
   - Initialize lists to store images, masks, star data, WCS data, and FITS file names.

3. **Importing the Dataset**:
   - Use the `importDataset` function to load the dataset and extract images, masks, star data, WCS data, and FITS file names.

4. **Preparing Images and Masks for the Model**:
   - Convert images to 3-channel format and masks to single-channel format.
   - Normalize the images using min-max normalization.

5. **Building the U-Net Model**:
   - Define and compile the U-Net model using specified hyperparameters.

6. **Splitting the Stacked Images and Masks**:
   - Split the dataset into training and validation sets using the `train_test_split` function.

7. **Training the Model**:
   - Train the U-Net model using the training dataset with early stopping to prevent overfitting.

8. **Saving the Model**:
   - Save the trained model and log the model details, including history, parameters, and saved model name.

9. **Evaluating the Model**:
   - Evaluate the model's performance on the validation dataset by calculating loss and accuracy metrics.

10. **Visualizing Results**:
    - Visualize the loss and accuracy along each epoch.



### Data Gathering

The `createStarDataset` function is a crucial part of our data preparation pipeline. It is responsible for generating and saving FITS files that contain both image data and star catalog data. These FITS files are then used to train our model.

#### Functionality of `createStarDataset`

The `createStarDataset` function performs the following steps:

1. **Directory Creation**:
   - Creates a new directory named `data` if it does not already exist. This directory will store the generated FITS files.

2. **Coordinate Generation**:
   - Generates random coordinates (RA and Dec) while avoiding the galactic plane to ensure a diverse set of sky regions.

3. **Image Data Fetching**:
   - Uses the `SkyView` service to fetch image data from the DSS survey for the generated coordinates. The image data is saved as a FITS file.

4. **Star Catalog Fetching**:
   - Uses the `Vizier` service to fetch star catalog data for the same coordinates. The star catalog data is appended to the FITS file as a binary table HDU.

5. **Pixel Mask Creation**:
   - Creates a pixel mask indicating the positions of stars in the image. The pixel mask is saved as an additional HDU in the FITS file.

6. **Star Overlay Plot**:
   - Generates a plot of the image with star positions overlaid. This plot is saved as an image file and then converted to FITS format, appended to the original FITS file.

#### Using `createStarDataset` to Prepare the Dataset

To prepare the dataset for training the model, we use the `createStarDataset` function with the parameter `'data'`. This generates a set of FITS files containing image data, star catalog data, and pixel masks. These files are stored in the `data/fits/` directory.

```python
# Generate training data
createStarDataset(catalog_type='II/246', iterations=20, file_path='data/fits/data/', filename='data', pixels=1024)
```

For validation purposes, we use the `createStarDataset` function with the filename parameter `'validate'`. This generates a separate set of FITS files for validation, ensuring that the files have the name `validate0.fits` for the `validateModel.ipynb` notebook.

```python
# Generate validation data
createStarDataset(catalog_type='II/246', iterations=50, file_path='data/fits/validate/', filename='validate', pixels=1024)
```

### Importing Images and Star Data from the Dataset

In this section, we will import the images, masks, and star data from our prepared dataset using the `importDataset` function. This function reads the FITS files from the specified directory and extracts the necessary data for training our model.

### Major Functionality of the `dataGathering` Module

The `dataGathering` module contains several important functions that facilitate the preparation and visualization of our dataset. Below, we provide an overview of the major functionalities:

1. **Data Extraction Functions**:
   - `extractImageArray`: Extracts image data from FITS files.
   - `extractPixelMaskArray`: Extracts pixel mask data from FITS files.
   - `extractStarCatalog`: Extracts star catalog data from FITS files.

2. **Data Import Function**:
   - `importDataset`: Imports the dataset by reading FITS files from a specified directory and extracting images, masks, and star data.

3. **Visualization Functions**:
   - `getImagePlot`: Generates a plot of the image data.
   - `getPixelMaskPlot`: Generates a plot of the pixel mask data.
   - `displayRawImage`: Displays the raw image data.
   - `displayRawPixelMask`: Displays the raw pixel mask data.
   - `displayImagePlot`: Displays the image plot.
   - `displayPixelMaskPlot`: Displays the pixel mask plot.
   - `displayPixelMaskOverlayPlot`: Displays an overlay plot of the image and pixel mask.

4. **Star Data Functions**:
   - `createStarDataset`: Generates and saves FITS files containing image data and star catalog data.
   - `importDataset`: Imports the dataset by reading FITS files from a specified directory and extracting images, masks, and star data.

These functions work together to streamline the process of preparing and visualizing our dataset, ensuring that we have high-quality data for training and validating our model.