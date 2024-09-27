# AstrID
AstrID: A project focused on identifying and classifying astronomical objects using data from various space catalogs. Leveraging machine learning, AstrID aims to enhance our understanding of stars, galaxies, and other celestial phenomena.

## Project Goals and Objectives
The primary goal of AstrID is to develop a robust system for identifying and classifying various astronomical objects, such as stars and galaxies, using data from space catalogs. A stretch goal of the project is to identify potential black hole candidates using advanced machine learning techniques.

## Features
- **Data Retrieval**: Fetch high-resolution images and data from space catalogs like Hipparcos and 2MASS.
- **Image Processing**: Process and visualize astronomical images with WCS overlays.
- **Machine Learning**: Classify different types of stars and other celestial objects.
- **Black Hole Identification**: (Stretch Goal) Identify potential black hole candidates.


## Instructions:

### IPYNB Installation
How to install the program and prepare for running.

Navigate to main folder, create a new virtual environment -			
    
    python3 -m venv .venv

Activate the environment - 				
    
    source .venv/bin/activate

Install required packages - 				
    
    pip install -r requirements.txt

---
Below section only required if viewing notebooks:

if viewing notebooks in VS Code - 

    - Install Python extension into VSCode

    - Install Jupyter extension into VSCode -

else … follow now your IDE procure

Install python kernel for your created venv - 
    
    python3 -m ipykernel install --user --name=.venv

Select your installed kernel to be used with your notebook - “Select Kernel” button in top right.
Alternatively notebooks may be viewed in any application capable of handling *.ipynb files

# Note: Ensure the following system dependency is installed:
    
    sudo apt-get install libgl1-mesa-glx