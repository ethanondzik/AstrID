To create a virtual environment using a specific Python version (3.10) on a server where you don't have admin rights and Python 3.10 is not available in 

bin

, you can follow these steps:

### Steps to Create a Virtual Environment with Python 3.10

1. **Download and Install Python 3.10 Locally**:
   - Download the Python 3.10 source code.
   - Compile and install Python 3.10 in your home directory.

2. **Create a Virtual Environment Using the Local Python 3.10**:
   - Use the locally installed Python 3.10 to create a virtual environment.

### Detailed Steps

#### 1. Download and Install Python 3.10 Locally

1. **Download Python 3.10 Source Code**:
   ```sh
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
   Use the locally installed Python 3.10 to create a virtual environment.

   ```sh
   python3.10 -m venv ~/myenv
   ```

3. **Activate the Virtual Environment**:
   Activate the virtual environment.

   ```sh
   source ~/myenv/bin/activate
   ```

### Example Commands Combined

Here are the commands combined for easy reference:

```sh
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

# Create a virtual environment using the local Python 3.10
python3.10 -m venv ~/myenv

# Activate the virtual environment
source ~/myenv/bin/activate
```

By following these steps, you should be able to create and use a virtual environment with Python 3.10 on a server without needing admin rights.