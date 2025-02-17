# Emotion Detection using CNN

## Overview
This project implements a **Facial Emotion Recognition System** using **Convolutional Neural Networks (CNNs)**. The system captures real-time facial expressions through a webcam and classifies emotions into **Happy, Sad, Neutral, and Angry**. It is primarily designed to enhance **customer satisfaction assessment in restaurants** by analyzing customers' facial expressions.

## Features
- **Real-time Emotion Detection** via webcam
- **Pre-trained CNN Model** for emotion classification
- **HaarCascade Classifier** for face detection
- **Dataset sourced from Kaggle & GitHub** with 26,217 images
- **Achieved Accuracy**: 74.82% on the validation dataset

## Project Structure
```
├── CVIP_Pdf.pdf             # Project report
├── README.md                # Project Documentation
├── emotiondetector.json     # Model architecture
├── realtimedetection.py     # Script for real-time emotion detection
├── trainmodel.ipynb         # Model training script
```

## Installation
### Prerequisites
Ensure you have **Python 3.7+** installed and install the required dependencies:
```bash
pip install numpy opencv-python tensorflow keras gdown
```

## Running the Project
To start the **real-time emotion detection**, run:
```bash
python realtimedetection.py
```

## Model Architecture
The model consists of **four convolutional layers**, each followed by:
- **MaxPooling** to reduce spatial dimensions
- **Dropout layers** to prevent overfitting
- **Flatten layer** to convert features into a vector
- **Fully connected dense layers** with **SoftMax activation** for classification

### Model Summary
| Layer Type         | Output Shape  | Parameters |
|--------------------|--------------|------------|
| Conv2D (128)      | (46, 46, 128) | 1,280      |
| MaxPooling2D      | (23, 23, 128) | 0          |
| Dropout (0.4)     | (23, 23, 128) | 0          |
| Conv2D (256)      | (21, 21, 256) | 295,168    |
| MaxPooling2D      | (10, 10, 256) | 0          |
| Dropout (0.4)     | (10, 10, 256) | 0          |
| Conv2D (512)      | (8, 8, 512)   | 1,180,160  |
| MaxPooling2D      | (4, 4, 512)   | 0          |
| Dropout (0.4)     | (4, 4, 512)   | 0          |
| Flatten           | (512)         | 0          |
| Dense (512)       | (512)         | 262,656    |
| Dropout (0.4)     | (512)         | 0          |
| Dense (256)       | (256)         | 131,328    |
| Dropout (0.3)     | (256)         | 0          |
| Dense (4)         | (4)           | 1,028      |

## Emotion Detection Process
1. Loads the trained model architecture (`emotiondetector.json`).
2. Captures live video frames from the webcam.
3. Converts the image to **grayscale** and applies **HaarCascade face detection**.
4. Resizes the detected face to **48x48 pixels** (model input size).
5. Passes the preprocessed image through the CNN for emotion classification.
6. Displays the **predicted emotion** on the video frame.

## Sample Output
Below are sample images showing real-time emotion detection:

### Angry
<img width="222" alt="image" src="https://github.com/user-attachments/assets/1c521f22-d53c-4f61-af1f-adc213f34f50" />


### Happy
<img width="224" alt="image" src="https://github.com/user-attachments/assets/6e09a14d-b6d4-4657-85b8-17a00f6abbb2" />


### Neutral
<img width="226" alt="image" src="https://github.com/user-attachments/assets/cd1ea301-0752-480d-a97f-8080b206b576" />

### Sad
<img width="227" alt="image" src="https://github.com/user-attachments/assets/f942ba55-67e8-49b6-b43a-80c4e2c75e47" />

## References
- OpenCV Face Detection: [OpenCV Docs](http://docs.opencv.org/master/d7/d8b/tutorialpyfacedetection.html)
- Deep Learning-based Emotion Recognition: Various research papers and GitHub implementations

## Author
**Akshay Pregada**
