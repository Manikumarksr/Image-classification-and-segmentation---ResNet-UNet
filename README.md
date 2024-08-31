# Plant Disease Classification and Segmentation

This repository contains code for plant disease binary classification using ResNet and segmentation using U-Net. The project is designed to help in identifying and segmenting diseased areas in plant leaves. This project is a part of 'Machine Learning for Data Science' course 

## Project Overview

1. **Binary Classification using ResNet**: 
   - This module classifies plant leaves into two categories: healthy and diseased.
   - The model is built using a ResNet architecture to leverage its deep feature extraction capabilities.

2. **Segmentation using U-Net**: 
   - This module segments the diseased regions in plant leaves.
   - U-Net architecture is used for precise localization and segmentation, making it well-suited for medical and biological imaging tasks.

## Data

The training data for both classification and segmentation can be found at the following link: [Training Data](https://drive.google.com/file/d/1PL8fFyHLQr4bNQy2ejF98g-igfA9K_KU/view?usp=drive_link)

Please ensure that you have downloaded and unzipped the dataset in the `/` directory before running the notebooks.

## Notebooks

- **`Binary_Classification_ResNet.ipynb`**:
  - Contains the code for training and evaluating the ResNet model for binary classification.

- **`Segmentation_UNet.ipynb`**:
  - Contains the code for training and evaluating the U-Net model for segmentation.

## Dependencies
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

## Usage
**Clone this repository:**
```
git clone https://github.com/Manikumarksr/Image-classification-and-segmentation---ResNet-UNet.git
cd Plant-Disease-Classification-Segmentation

```
Open the Jupyter notebooks and run the cells sequentially.

## Results
The models are evaluated based on accuracy for classification and DICE score for segmentation. Example outputs are included in the notebooks.
