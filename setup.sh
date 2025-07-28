#!/bin/bash

git clone git@github.com:ykerverdo/app_seg.git
cd app_seg

mkdir atlas
cd atlas
wget https://github.com/Inria-Empenn/Anima-Scripts-Data-Public/raw/refs/heads/master/icc_atlas/Reference_T1.nrrd
wget https://github.com/Inria-Empenn/Anima-Scripts-Data-Public/raw/refs/heads/master/icc_atlas/BrainMask.nrrd
cd ..


wget https://github.com/Inria-Empenn/Anima-Public/releases/download/v4.2/Anima-Fedora-4.2.zip
if ! command -v unzip
then
  echo "unzip non trouvé, installation..."
  sudo dnf install -y unzip
fi
unzip Anima-Fedora-4.2.zip
mkdir anima
cp ./Anima-Binaries-4.2/animaApplyTransformSerie ./anima/
cp ./Anima-Binaries-4.2/animaConvertImage ./anima/
cp ./Anima-Binaries-4.2/animaCreateImage ./anima/
cp ./Anima-Binaries-4.2/animaDenseSVFBMRegistration ./anima/
cp ./Anima-Binaries-4.2/animaMaskImage ./anima/
cp ./Anima-Binaries-4.2/animaN4BiasCorrection ./anima/
cp ./Anima-Binaries-4.2/animaPyramidalBMRegistration ./anima/
cp ./Anima-Binaries-4.2/animaTransformSerieXmlGenerator ./anima/
rm Anima-Fedora-4.2.zip
rm -rf Anima-Binaries-4.2/

if ! command -v python3
then
  echo "Python3 non trouvé, installation..."
  sudo dnf install -y python3
fi
python3 -m venv venv 
source venv/bin/activate
pip install --upgrade pip
pip install onnxruntime[cudnn] nibabel scipy batchgenerators