# Download and extract the BZip2 source code
wget https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz
tar -xvf bzip2-1.0.8.tar.gz
cd bzip2-1.0.8

# Compile and install BZip2 locally
make
make PREFIX=$HOME/local install

# Download and extract the Python source code
wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
tar -xvf Python-3.10.0.tgz
cd Python-3.10.0

# Configure the Python build to use the local libraries
LDFLAGS="-L$HOME/local/lib" CPPFLAGS="-I$HOME/local/include" ./configure --prefix=$HOME/localpython

# Compile and install Python
make
make install

# Add the local Python to your PATH
export PATH=$HOME/localpython/bin:$PATH