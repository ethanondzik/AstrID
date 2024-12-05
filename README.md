# AstrID
*AstrID:* A project focused on identifying and classifying astronomical objects using data from various space catalogs. Leveraging machine learning, AstrID aims to enhance our understanding of stars, galaxies, and other celestial phenomena.

## Project Goals and Objectives
The primary goal of AstrID is to develop a robust system for identifying the location of astronomical objects, primarily stars, in images of space taken by terrestrial and satellite radio-telescopes. A stretch goal of the project is to identify potential black hole candidates using advanced machine learning techniques.

## Features
- **Data Retrieval**: Fetch high-resolution images and data from space catalogs like Hipparcos and 2MASS.
- **Image Processing**: Process and visualize astronomical images with WCS overlays.
- **Machine Learning**: Classify different types of stars and other celestial objects using a U-Net model.
- **Model Training**: Train machine learning models using high-resolution astronomical images.
- **Model Evaluation**: Evaluate the performance of trained models on validation and test datasets.
- **Prediction**: Make predictions on new astronomical data using trained models.
- **Black Hole Identification**: (Stretch Goal) Identify potential black hole candidates.

## Results 

### The Dataset
Our team was unable to find a pre-compiled dataset that met our needs. Instead, we set out to compile our own dataset for use in this project. Using the AstroPy library, we were able to query both the SkyView and Vizier databases. Using SkyView, we were able to download a random set of 250 images from across the sky, by randomly generating world-coordinates and pulling images from that location. For each image, we then queried the Vizier star catalog at the same coordinates, downloading a list of which stars were present in each image, and what their world-coordinates were. 

By converting from the world coordinates to the corresponding coordinates in the SkyView image, we were able to generate a 'mask' or the ground truth image indicating which stars were where. Using the .fits image format, we were able to save the original image, the mask, and the star data table in the same file.

Our FITS Components:

1. Primary HDU (Header Data Unit)
    - Contains the main image data and associated metadata (2D array of pixel values)

2. Star Catalog HDU:
    - Information about stars in the image, e.g, coordinates and magnitudes. (binary table)

3. Pixel Mask HDU:
    - Image containing a pixel mask/ground truth indicating the positions of stars in the image. (2D array of pixel values where pixels corresponding to stars are set to 1)

An example of this can be seen in the image below, where a random image from SkyView is overlaid with the stars in that region listed in the Vizier catalog. Each blue dot indicates the location of a star, according to the vizier catalog.  

![output](https://github.com/user-attachments/assets/ff1cbc73-7de3-4342-a5ae-0800b48c0504)


### Model and Training
Our team opted to use a Convolutional Neural Network (CNN) model to make predictions, specifically a U-Net model. We implemented this model using the Keras library, which made the initial programming simpler than a more robust library, but later caused difficulties when we tried more advanced functionality such as implementing class weights or using more bespoke accuracy metrics. We experimented with a variety of hyper-paremeters, and implemented a logging functionality that recorded the results of each training run. This allowed us to quickly identify the best performing models when tuning the model's performance. We found that learning rate and class weights had the most impact on the performance of the model, but we experimented with tuning other parameters as well. 

![image](https://github.com/user-attachments/assets/90c17837-e986-462c-8cb9-232c0000aaeb)

Our model did take considerable time to train using a CPU, especially as running 30+ epochs seems to be the optimal performance window. Some of our team members were able to run the training on a GPU, which considerably sped up the training process and is recommended. We took additional steps to optimize the training process, such as using early-stopping, which would stop training if no improvement in the model was observed after 10 consecutive training epochs. Given more time, we would have liked to experiment with other metrics for performance, such as f-1 score. 

![image](https://github.com/user-attachments/assets/599f839c-a268-4c7a-b93e-2a38009bdcfd)
![image](https://github.com/user-attachments/assets/0242d8ea-fb3a-4a71-af8c-3463bbdbe4e4)


### Predictions
Our final model was able to predict the location of stars with reasonable accuracy. In the below image, the blue circles indicate the known-location of stars from the Vizier catalog, and the red circles indicate our models' predicted locations of stars. Our model does struggle with the interpretation of photographic anomalies, such as the lens flare visible in the below image. 

However, anomalous images were sparse in our dataset, and training on a dataset with more images or even identifying the anomalies in our ground-truth mask may help reduce the false prediction of stars in these locations. Interestingly, our model was able to pick up on stars and star-like objects present in the image, but not listed in the Vizier catalog. This indicates that our model is performing well, and may provide a useful analysis of images from uncataloged regions of space. 

![StarOverlay](https://github.com/user-attachments/assets/0ceaa54a-ccb7-4c15-8036-89cc033983ff)
![2024_12_05-093117_chris_gray_r_predictions_overlay](https://github.com/user-attachments/assets/2030079b-88dc-4a76-bbb5-08b270209ac0)

Our team found a substantial improvement in the prediction accuracy when we account for the intensity (vmag value) of a star when generating the ground-truth, or 'mask'. This can be observed in the below image, where the locations of stars indicated in the 'mask' are of varying sizes. Previously, the locations of stars were indicated with a single pixel, which caused our model to have problems identifying larger stars in the image. 

![validate3](https://github.com/user-attachments/assets/8e4e5480-506f-4d54-849b-b9fa624ca4c1)
![validate3overlay](https://github.com/user-attachments/assets/5933d098-c27d-403b-8b99-e36c3ac02199)


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
