# AstrID
*AstrID:* A project focused on identifying and classifying astronomical objects using data from various space catalogs. 
Leveraging machine learning, AstrID aims to enhance our understanding of stars, galaxies, and other celestial phenomena.

## Project Goals and Objectives
The primary goal of AstrID is to develop a robust system for identifying and classifying various astronomical objects, such as stars and galaxies, using data from space catalogs. 
A stretch goal of the project is to identify potential black hole candidates using advanced machine learning techniques.

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

Ensure the following system dependency is installed:
```bash
sudo apt-get install libgl1-mesa-glx