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