# Deepfake Detection using Convolutional Neural Networks
ECE 5831 — Pattern Recognition and Neural Networks — Final Project

## Overview:

This project presents a deepfake detection system that classifies face images as real or fake using a custom convolutional neural network (CNN) called RobustCNN. The model was trained on a curated deepfake dataset consisting of equal real and manipulated facial frames, that consists of three different versions including DeepFake, FaceSwap, and Face2Face manipulations.

The goal of this project is to build an efficient and interpretable deepfake detector capable of identifying visual manipulation artifacts at the image level. The project includes:

Dataset preprocessing (frame extraction, optional face cropping)

RobustCNN architecture implementation

Training & validation on GPU (Google Colab)

Exporting the trained model (best_model.pth)

Test images

Report

Presentation video 

## Links:
YouTube Presentation Video: 
Slides (Google Drive): https://docs.google.com/presentation/d/1SK_hj1sRqKiampcqTxsHaIeCfozC0jV9/edit?slide=id.p1#slide=id.p1
Final Report (PDF): https://drive.google.com/file/d/1wiSNd2djxwD4RmcV7dp-gnR2Unv1jXQi/view
Dataset Folder: https://drive.google.com/drive/folders/13wcCdnD9NGdefKIAz7gXFBHJmXU8iMUJ

google drive: https://drive.google.com/drive/folders/1VZpehDG5WkFIrAhF1cYkNyrzD0_-SDZN?usp=sharing

## Dataset Description
The dataset was created by extracting frames from real and manipulated videos. Preprocessing steps include:

- Frame extraction at fixed intervals
- Face detection using MTCNN
- Resizing to 128×128
- Normalization to [-1, 1]
- Balanced sampling for train(80%)/val(20%)
- 
## Model Architecture (RobustCNN)
The model consists of four convolutional blocks followed by two fully connected layers:
- Convolutional Blocks
- Conv2D → BatchNorm → LeakyReLU → MaxPool
- Channels: 3 → 32 → 64 → 128 → 256
- Resolution reduces: 128→64→32→16→8
Fully Connected
- Flatten → Dense(16384 → 512) → Dropout
- Output: 2 neurons (real, fake)

## Results
Validation Accuracy: 93.92%

## Challenges Encountered
- The Google Drive I/O Bottleneck
- The "Majority Class" Bias
- The "Black Image" Bug
- GPU training time limits
- Model Convergence Failure
