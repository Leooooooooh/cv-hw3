#!/bin/bash

# Name of the conda environment
ENV_NAME="hw3_seg"

# Create conda environment with Python 3.9
echo "Creating conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME python=3.9

# Activate the environment
echo "Activating conda environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install PyTorch (Change the CUDA version if needed)
echo "Installing PyTorch"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required packages
echo "Installing other dependencies"
pip install \
  opencv-python \
  matplotlib \
  tqdm \
  scikit-image \
  pycocotools \
  jupyter

# Freeze dependencies to requirements.txt
echo "Saving requirements.txt"
pip freeze > requirements.txt

echo "✅ Environment setup complete. Use 'conda activate $ENV_NAME' to start working."