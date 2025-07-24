#!/bin/bash

# Exit on any error
set -e

echo "Starting installation of Ergonomic Assessment Backend"
echo "======================================================"

# Check current Python version
CURRENT_PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
echo "Current Python version: $CURRENT_PYTHON_VERSION"

# Install Python 3.9 if not available or if current version is 3.8
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]; then
    echo "Python 3.9+ required. Installing Python 3.9 with SSL support..."
    
    # Update package list
    echo "[1/10] Updating package list..."
    sudo apt-get update -y
    
    # Install build dependencies
    echo "[2/10] Installing build dependencies..."
    sudo apt-get install -y \
        build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        wget \
        libbz2-dev \
        software-properties-common
    
    # Download and compile Python 3.9 from source with SSL support
    echo "[3/10] Downloading Python 3.9.18 source..."
    cd /tmp
    wget https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tgz
    tar -xf Python-3.9.18.tgz
    cd Python-3.9.18
    
    echo "[4/10] Configuring Python build with SSL..."
    ./configure --enable-optimizations --with-ssl-default-suites=openssl --enable-loadable-sqlite-extensions
    
    echo "[5/10] Compiling Python 3.9 (this may take a while)..."
    make -j$(nproc)
    
    echo "[6/10] Installing Python 3.9..."
    sudo make altinstall
    
    # Clean up
    cd /
    rm -rf /tmp/Python-3.9.18*
    
    # Set Python 3.9 as the default python3
    echo "[7/10] Setting Python 3.9 as default..."
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.9 2
    
    # Automatically select Python 3.9
    echo "2" | sudo update-alternatives --config python3
    
    # Create symlink for pip3
    sudo ln -sf /usr/local/bin/pip3.9 /usr/bin/pip3
    
    echo "Python 3.9 compiled and installed with SSL support"
else
    echo "Python 3.9+ already available"
    # Update package list
    echo "[1/10] Updating package list..."
    sudo apt-get update -y
fi

# Verify Python version and SSL support
NEW_PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
echo "Using Python version: $NEW_PYTHON_VERSION"

# Test SSL support
echo "[8/10] Testing SSL support..."
python3 -c "import ssl; print('SSL support: OK')" || echo "SSL support: FAILED"

# Install system dependencies
echo "[9/10] Installing system dependencies..."
sudo apt-get install -y --no-install-recommends \
    libopencv-dev \
    python3-opencv \
    ffmpeg \
    git

# Install Python dependencies
echo "[10/10] Installing Python dependencies..."
python3 -m pip install --upgrade pip setuptools wheel

python3 -m pip install \
    flask \
    flask-socketio \
    flask-cors \
    eventlet \
    numpy \
    pandas \
    matplotlib \
    tensorflow \
    tensorflow-hub \
    scikit-learn \
    joblib \
    opencv-python \
    imageio \
    tqdm \
    scipy

echo "All dependencies installed successfully"

# Create required directories
mkdir -p temp_jobs
mkdir -p output_images
mkdir -p logs
mkdir -p modelv4
mkdir -p movenet_models

# Set up TensorFlow cache directory
chmod -R 755 temp_jobs output_images logs modelv4 movenet_models
mkdir -p ~/.cache/tensorflow-hub
chmod -R 777 ~/.cache/tensorflow-hub

# Clone repository if not already present
echo "Setting up repository..."
if [ ! -d "TA_Deployment" ]; then
    git clone https://github.com/Informatics-ITS/ta-HADAIZI.git
    echo "Repository cloned"
else
    echo "Repository already exists"
fi

echo "======================================================"
echo "Installation complete!"
echo "Python version: $(python3 --version)"
echo "SSL test: $(python3 -c 'import ssl; print("SSL working!")' 2>/dev/null || echo 'SSL failed')"
echo "You can now start the server by running: python3 run.py"
echo "The server will be available at http://localhost:5050"
cd ta-HADAIZI