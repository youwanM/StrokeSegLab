#!/bin/bash

git clone git@github.com:ykerverdo/app_seg.git
cd app_seg

wget https://github.com/Inria-Empenn/Anima-Public/releases/download/v4.2/Anima-Fedora-4.2.zip
if ! command -v unzip
then
  echo "unzip non trouvé, installation..."
  sudo dnf install -y unzip
fi
unzip Anima-Fedora-4.2.zip
mkdir anima
cp ./Anima-Fedora-4.2/animaApplyTransformSerie ./anima/
cp ./Anima-Fedora-4.2/animaConvertImage ./anima/
cp ./Anima-Fedora-4.2/animaCreateImage ./anima/
cp ./Anima-Fedora-4.2/animaDenseSVFBMRegistration ./anima/
cp ./Anima-Fedora-4.2/animaMaskImage ./anima/
cp ./Anima-Fedora-4.2/animaN4BiasCorrection ./anima/
cp ./Anima-Fedora-4.2/animaPyramidalBMRegistration ./anima/
cp ./Anima-Fedora-4.2/animaTransformSerieXmlGenerator ./anima/
rm Anima-Fedora-4.2.zip

if ! command -v python3
then
  echo "Python3 non trouvé, installation..."
  sudo dnf install -y python3
fi
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install onnxruntime[cudnn] nibabel scipy batchgenerators