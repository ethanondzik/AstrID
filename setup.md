### Step-by-Step Guide

#### 1. Install Python 3.10 and Virtualenv

Ensure that Python 3.10 and `virtualenv` are installed on your system.

```bash
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv python3.10-dev
pip install virtualenv
```

#### 2. Create a Virtual Environment

Create a new virtual environment using `virtualenv`.

```bash
virtualenv --python=/usr/bin/python3.10 .venv
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

#### 4. Install CUDA 11.8

1. **Download CUDA 11.8**:
   - Download the CUDA 11.8 installer from the [NVIDIA CUDA Toolkit website](https://developer.nvidia.com/cuda-11-8-0-download-archive).

2. **Install CUDA 11.8**:
   - Follow the installation instructions provided on the NVIDIA website. Here is an example of how to install CUDA 11.8:

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
   sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
   sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-9EA88183-keyring.gpg /usr/share/keyrings/
   sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/7fa2af80.pub
   sudo apt-get update
   sudo apt-get install -y cuda-11-8
   ```

#### 5. Install cuDNN 8.6

1. **Download cuDNN 8.6**:
   - Download the cuDNN 8.6 library for CUDA 11.8 from the [NVIDIA cuDNN website](https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz).

2. **Extract the cuDNN Archive**:
   - Extract the downloaded cuDNN archive.

   ```bash
   tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
   ```

3. **Copy the cuDNN Files**:
   - Copy the extracted files to the appropriate CUDA directories.

   ```bash
   sudo cp cudnn-linux-x86_64-8.6.0.163_cuda11-archive/include/cudnn*.h /usr/local/cuda-11.8/include
   sudo cp -P cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64/
   sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
   ```

#### 6. Set Environment Variables

Ensure that the environment variables for CUDA 11.8 and cuDNN 8.6 are correctly set in your `.bashrc` or `.bash_profile`.

```bash
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 7. Verify CUDA and cuDNN Installation

Verify the installation of CUDA and cuDNN.

```bash
nvcc --version
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

#### 8. Test TensorFlow GPU Support

Run a simple TensorFlow script to check if TensorFlow can detect the GPU.

```python
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

### Example Commands Combined

Here are the commands combined for easy reference:

```bash
# Install Python 3.10 and virtualenv
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv python3.10-dev
pip install virtualenv

# Create and activate a virtual environment
virtualenv --python=/usr/bin/python3.10 .venv
source .venv/bin/activate

# Install TensorFlow with GPU support
pip install tensorflow

# Uninstall existing CUDA and cuDNN versions
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" "nvidia*"

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
```