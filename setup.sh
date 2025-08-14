#!/bin/bash

app_name="StrokeSeg"

# Detect if system is Fedora (dnf) or Debian/Ubuntu (apt)
if command -v dnf >/dev/null 2>&1; then
    IS_FEDORA=true
elif command -v apt-get >/dev/null 2>&1; then
    IS_FEDORA=false
else
    echo "Unsupported system. Please install required packages manually"
    exit 1
fi

# Create a base directory
mkdir -p "$app_name"
echo "Base direcory '$app_name' created"
cd "$app_name"

# Clone the app repository
if [ ! -d "app_seg" ]; then
  git clone git@github.com:ykerverdo/app_seg.git
  echo "app repository cloned"
else
  cd app_seg && git pull
  echo "Directory 'app_seg' already exists, updated with latest changes"
  cd ..
fi

# Check if wget is installed, install if missing
if ! command -v wget >/dev/null 2>&1; then
  echo "wget not found, installing..."
  if [ "$IS_FEDORA" = true ]; then
    sudo dnf install -y wget
  else
    sudo apt update && sudo apt install -y wget
  fi
else
  echo "wget is already installed"
fi

# Create 'atlas' directory and download files if missing
mkdir -p atlas
cd atlas

if [ ! -f "Reference_T1.nrrd" ]; then
  echo "Downloading Reference_T1.nrrd..."
  wget https://github.com/Inria-Empenn/Anima-Scripts-Data-Public/raw/refs/heads/master/icc_atlas/Reference_T1.nrrd
else
  echo "Reference_T1.nrrd already exists, skipping download"
fi

if [ ! -f "BrainMask.nrrd" ]; then
  echo "Downloading BrainMask.nrrd..."
  wget https://github.com/Inria-Empenn/Anima-Scripts-Data-Public/raw/refs/heads/master/icc_atlas/BrainMask.nrrd
else
  echo "BrainMask.nrrd already exists, skipping download"
fi

cd ..
echo "'atlas' setup complete"


# Check if unzip is installed, install if missing
if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip not found, installing..."
  if [ "$IS_FEDORA" = true ]; then
    sudo dnf install -y unzip
  else
    sudo apt update && sudo apt install -y unzip
  fi
else
  echo "unzip is already installed"
fi

# Ensure 'anima' directory exists
mkdir -p anima

# Download Anima zip if not already present
if [ ! -f "Anima-Fedora-4.2.zip" ]; then
  echo "Downloading Anima-Fedora-4.2.zip..."
  wget https://github.com/Inria-Empenn/Anima-Public/releases/download/v4.2/Anima-Fedora-4.2.zip
else
  echo "Anima-Fedora-4.2.zip already exists, skipping download"
fi

# Unzip if binaries folder does not exist
if [ ! -d "Anima-Binaries-4.2" ]; then
  echo "Unzipping Anima-Fedora-4.2.zip..."
  unzip Anima-Fedora-4.2.zip
else
  echo "'Anima-Binaries-4.2' already exists, skipping unzip"
fi

# List of binaries to copy
binaries=(
  "animaApplyTransformSerie"
  "animaConvertImage"
  "animaCreateImage"
  "animaDenseSVFBMRegistration"
  "animaMaskImage"
  "animaN4BiasCorrection"
  "animaPyramidalBMRegistration"
  "animaTransformSerieXmlGenerator"
)

# Copy binaries if not already in 'anima'
for bin in "${binaries[@]}"; do
  if [ ! -f "anima/$bin" ]; then
    echo "Copying $bin..."
    cp "./Anima-Binaries-4.2/$bin" ./anima/
  else
    echo "$bin already exists in 'anima', skipping"
  fi
done

# Cleanup zip if binaries were extracted
if [ -d "Anima-Binaries-4.2" ]; then
  echo "Cleaning up zip and temporary folder..."
  rm Anima-Fedora-4.2.zip
  rm -rf Anima-Binaries-4.2/
fi

echo "Anima setup complete"


# Check if python3 is installed, install if missing
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python3 not found, installing..."
  if [ "$is_fedora" = true ]; then
    sudo dnf install -y python3
  else
    sudo apt install -y python3
  fi
else
  echo "Python3 is already installed"
fi

cd app_seg
# Remove existing virtual environment if present
if [ -d "venv" ]; then
  echo "Removing existing virtual environment..."
  rm -rf venv
fi

# Create a new virtual environment
echo "Creating new virtual environment..."
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip and install required packages
echo "Upgrading pip and installing dependencies..."
pip install --upgrade pip
# Check if CUDA is available
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "CUDA detected, installing GPU version of onnxruntime..."
  pip install --upgrade pip
  pip install "onnxruntime-gpu[cudnn]" nibabel scipy batchgenerators
else
  echo "CUDA not detected, installing CPU version of onnxruntime..."
  pip install --upgrade pip
  pip install onnxruntime nibabel scipy batchgenerators
fi

echo "Virtual environment setup complete"