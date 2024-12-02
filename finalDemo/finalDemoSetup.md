### Steps to Create a Virtual Environment with Python 3.10

To create a virtual environment using a specific Python version (3.10) on a server where you don't have admin rights and Python 3.10 is not available in `bin`, you can follow these steps:

1. **Download and Install Python 3.10 Locally**:
   - Download the Python 3.10 source code.
   - Compile and install Python 3.10 in your home directory.

2. **Create a Virtual Environment Using the Local Python 3.10**:
   - Use the locally installed Python 3.10 to create a virtual environment.

### Detailed Steps

#### 1. Download and Install Python 3.10 Locally

1. **Download Python 3.10 Source Code**:
   ```sh
    cd finalDemo

    wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
   ```

2. **Extract the Tarball**:
   ```sh
    tar -xvf Python-3.10.0.tgz
    cd Python-3.10.0
   ```

3. **Configure the Build**:
   Configure the build to install Python in your home directory (e.g., `~/localpython`).

   ```sh
    ./configure --prefix=$HOME/localpython
   ```

4. **Compile and Install**:
   ```sh
    make
    make install
   ```

   This will install Python 3.10 in `~/localpython/bin`.

#### 2. Create a Virtual Environment Using the Local Python 3.10

1. **Add the Local Python to Your PATH**:
   Add the local Python installation to your PATH.

   ```sh
    export PATH=$HOME/localpython/bin:$PATH
   ```

   You can add this line to your `~/.bashrc` or `~/.bash_profile` to make it permanent.

2. **Create a Virtual Environment**:
   Navigate to the `finalDemo` folder and use the locally installed Python 3.10 to create a virtual environment.

    ```sh
    cd ..

    python3.10 -m venv .venv
   ```

3. **Activate the Virtual Environment**:
   Activate the virtual environment.

   ```sh
    source .venv/bin/activate
   ```

4. **Install Requirements**:
   Install the requirements with no cache to save quota.

   ```sh
    pip install --no-cache-dir -r requirements.txt
   ```

### Example Commands Combined

Here are the commands combined for easy reference. Execute these commands from the 

finalDemo

 folder:

```sh
# CD into the finalDemo folder
cd finalDemo

# Download Python 3.10 source code
wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz

# Extract the tarball
tar -xvf Python-3.10.0.tgz
cd Python-3.10.0

# Configure the build to install Python in your home directory
./configure --prefix=$HOME/localpython

# Compile and install
make
make install

# Add the local Python to your PATH
export PATH=$HOME/localpython/bin:$PATH

# CD into the parent directory finalDemo
cd ..

# Create a virtual environment using the local Python 3.10
python3.10 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the requirements
pip install --no-cache-dir -r requirements.txt

# Run the demo script
python3 demoModel.py finalDemo/image.png 
```

By following these steps, you should be able to create and use a virtual environment with Python 3.10 on a server without needing admin rights.