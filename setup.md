# Setup Guide

This guide will help you set up the AstrID project on your local machine.

## Prerequisites

- Python 3.6 or higher
- pip (Python package installer)
- Virtual environment tool (optional but recommended)

## Installation Steps

### 1. Clone the Repository

Clone the AstrID repository from GitHub to your local machine:

```sh
git clone https://github.com/yourusername/AstrID.git
cd AstrID
```

### 2. Create a Virtual Environment

It is recommended to create a virtual environment to manage dependencies:

```sh
python3 -m venv .venv
```

### 3. Activate the Virtual Environment

Activate the virtual environment:

- On Windows:
  ```sh
  .venv\Scripts\activate
  ```
- On macOS/Linux:
  ```sh
  source .venv/bin/activate
  ```

### 4. Install Required Packages

Install the required packages using `pip`:

```sh
pip install -r requirements.txt
```

### 5. Install System Dependencies

Ensure the following system dependency is installed:

```sh
sudo apt-get install libgl1-mesa-glx
```

### 6. Set Up Jupyter Notebook (Optional)

If you plan to use Jupyter Notebook, install the Jupyter kernel for your virtual environment:

```sh
python3 -m ipykernel install --user --name=.venv
```

### 7. Configure VS Code (Optional)

If you are using VS Code, install the Python and Jupyter extensions. Select the installed kernel to be used with your notebook by clicking the "Select Kernel" button in the top right.

## You're All Set!

You are now ready to start working on the AstrID project. Refer to the `usage.md` file for instructions on how to use the project.