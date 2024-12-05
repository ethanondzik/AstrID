# **AstrID**: *finalDemo*

### Final Demo Setup Instructions

This document provides step-by-step instructions to set up and run the final demo for our model. Follow these instructions to create a virtual environment, install the necessary dependencies, and run the demo script.

#### Prerequisites

Ensure you have **Python 3.11** in your system to run this demo. (alternatively Python3.10, cuDNN 8.6 and CUDA 11.8 required for GPU support. Instructions in docs/GPU_setup.md)

### Step 1: Create a Virtual Environment

1. **Navigate to the Project Directory**:
   Open a terminal and navigate to the `finalDemo` directory of the project.

   ```sh
   cd path/to/AstrID/finalDemo
   ```

2. **Create a Virtual Environment**:
   Use Python to create a virtual environment.

   ```sh
   python3 -m venv .venv
   ```

3. **Activate the Virtual Environment**:
   Activate the virtual environment.

   ```sh
   source .venv/bin/activate
   ```

4. **Install the Requirements**:
   Install the required dependencies using the `requirements.txt` file. (no-cache-dir option if you are on a tight quota!!)

   ```sh
   pip install --no-cache-dir -r requirements.txt
   ```

### Step 2: Run the Demo Script

1. **Activate the Virtual Environment**:
   Ensure the virtual environment is activated.

   ```sh
   source .venv/bin/activate
   ```

2. **Run the Demo Script**:
   Execute the `demoModel.py` script to run the model and generate predictions.

   ```sh
   python3 demoModel.py
   ```

### Step 3: Observe the Results

1. **Check the Predictions**:
   The script will generate prediction images and save them in the `predictions` directory. You can find the following types of images:

   - **Prediction Comparison**: Images comparing the original image, pixel mask, and model prediction mask.
   - **Prediction Overlay**: Images with star location and star prediction overlays.


2. **Navigate to the Predictions Directory**:
   Open the `predictions` directory to view the saved prediction images.

   ```sh
   cd predictions
   ```

3. **View the Images**:
   You can use any image viewer to open and observe the prediction images. The images will be saved with timestamps and user information for easy identification.

### Example Commands Combined

Here are the commands combined for easy reference. Execute these commands from the `finalDemo` directory:

```sh
# Create a virtual environment using the local Python 3.10
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the requirements
pip install --no-cache-dir -r requirements.txt

# Run the demo script
python3 demoModel.py

# Navigate to the predictions directory to view the results
cd predictions
```

By following these steps, you should be able to set up the environment, run the model demo, and observe the prediction results. If you encounter any issues, please ensure that all dependencies are correctly installed and that the virtual environment is activated.