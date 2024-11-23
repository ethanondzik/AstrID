### Step-by-Step Guide

#### 1. Install Python 3.10 and Virtualenv

Ensure that Python 3.10 and `virtualenv` are installed on your system.

```bash
sudo apt-get update
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
sudo apt install -y python3-pip
```

#### 2. Create a Virtual Environment

Create a new virtual environment using the built-in `venv` module.

```bash
python3.10 -m venv .venv
```

Activate the virtual environment.

```bash
source .venv/bin/activate
```

#### 3. Install TensorFlow with GPU Support

Install the GPU version of TensorFlow.

```bash
pip install tensorflow
```

#### 4. Uninstall Existing CUDA and cuDNN Versions

Uninstall any existing CUDA and cuDNN versions to avoid conflicts.

```bash
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" "nvidia*"
```

#### 5. Install libtinfo5

Install the `libtinfo5` library.

```bash
sudo apt-get install -y libtinfo5
```

#### 6. Install NVIDIA Driver

1. **Add NVIDIA PPA**:
    ```bash
    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt-get update
    ```

2. **Install the NVIDIA Driver**:
    ```bash
    sudo apt-get install -y nvidia-driver-525
    ```

3. **Reboot Your System**:
    ```bash
    sudo reboot
    ```

#### 7. Install CUDA 11.8

1. **Download CUDA 11.8**:
    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
    ```

2. **Install CUDA 11.8**:
    ```bash
    sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
    sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-9EA88183-keyring.gpg /usr/share/keyrings/
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install -y cuda-11-8
    ```

#### 8. Install cuDNN 8.6

1. **Download cuDNN 8.6**:
    ```bash
    wget https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
    ```

2. **Extract the cuDNN Archive**:
    ```bash
    tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
    ```

3. **Copy the cuDNN Files**:
    ```bash
    sudo cp cudnn-linux-x86_64-8.6.0.163_cuda11-archive/include/cudnn*.h /usr/local/cuda-11.8/include
    sudo cp -P cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64/
    sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
    ```

#### 9. Set Environment Variables

Ensure that the environment variables for CUDA 11.8 and cuDNN 8.6 are correctly set in your `.bashrc` or `.bash_profile`.

```bash
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 10. Verify CUDA and cuDNN Installation

Verify the installation of CUDA and cuDNN.

```bash
nvcc --version
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

#### 11. Test TensorFlow GPU Support

Run a simple TensorFlow script to check if TensorFlow can detect the GPU.

```python
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

#### 12. Resolve Dependency Conflicts

There is a dependency conflict between TensorFlow and the `typing_extensions` library, which requires a specific version to be installed.

1. **Create a constraints file to specify the version of `typing_extensions`**:
    ```bash
    echo "typing_extensions==4.5.0" > constraints.txt
    echo "ipykernel==6.29.5" >> constraints.txt
    echo "ipython==8.12.0" >> constraints.txt
    ```

2. **Uninstall the current `typing_extensions`**:
    ```bash
    pip uninstall -y typing_extensions
    ```

3. **Install the specific version of `typing_extensions` to ensure compatibility with TensorFlow**:
    ```bash
    pip install typing_extensions==4.5.0
    ```

4. **Reinstall `ipykernel` to ensure compatibility using constraints**:
    ```bash
    pip install --force-reinstall ipykernel -c constraints.txt
    ```

5. **Install the IPython kernel for the virtual environment**:
    ```bash
    python3 -m ipykernel install --user --name=.venv
    ```

### Example Commands Combined

To complete the above steps you can run [`setup.sh`](setup.sh) in order to complete the setup process in your Linux terminal.

Here are the commands combined for easy reference:

```bash
# Install Python 3.10 and virtualenv
sudo apt-get update
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
sudo apt install -y python3-pip

# Create and activate a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install TensorFlow with GPU support
pip install tensorflow

# Uninstall existing CUDA and cuDNN versions
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" "nvidia*"

# Install libtinfo5
sudo apt-get install -y libtinfo5

# Add NVIDIA PPA and install the NVIDIA driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install -y nvidia-driver-525

# Reboot the system to apply the NVIDIA driver installation
sudo reboot

# After reboot, continue with the following steps:

# Download and install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-9EA88183-keyring.gpg /usr/share/keyrings/
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y cuda-11-8

# Download and install cuDNN 8.6
wget https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.6.0.163_cuda11-archive/include/cudnn*.h /usr/local/cuda-11.8/include
sudo cp -P cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64/
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*

# Set environment variables for CUDA 11.8 and cuDNN 8.6
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA and cuDNN installation
nvcc --version
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

# Test TensorFlow GPU support
python -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"

# Resolve dependency conflicts
echo "typing_extensions==4.5.0" > constraints.txt
echo "ipykernel==6.29.5" >> constraints.txt
echo "ipython==8.12.0" >> constraints.txt

pip uninstall -y typing_extensions
pip install typing_extensions==4.5.0
pip install --force-reinstall ipykernel -c constraints.txt
python3 -m ipykernel install --user --name=.venv
```

By following these instructions, you should be able to set up your environment, view notebooks, and handle large files effectively.