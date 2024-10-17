# Real-Time Emotion Recognition

This project implements a **real-time emotion recognition system** using facial expressions. It is developed in both **Python** (using TensorFlow) and **MATLAB**. The system first trains a model on the **FER2013 dataset** to classify various emotions and then uses the trained model for live video-based emotion recognition through a camera. This system has potential applications in human-computer interaction, mental health analysis, and AI-driven applications.

## Table of Contents

- [Introduction](#introduction)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)

## Introduction

Facial expressions are a powerful way of conveying emotions. This project aims to build a system that can detect emotions from a live camera feed in real-time. By training a model on the widely-used FER2013 dataset, the system is able to classify emotions such as happiness, sadness, anger, fear, disgust, surprise, and neutrality.

The project is implemented in two programming environments:
1. **Python (TensorFlow)** - The primary system that has been trained and tested.
2. **MATLAB** - A secondary version for emotion recognition, though not fully tested due to compilation time constraints.

## Technologies

The project uses the following technologies:

### Python
- **TensorFlow**: For building and training the neural network model.
- **OpenCV**: For handling video capture and image processing.
- **NumPy**: For handling arrays and numerical data.
- **Keras**: For creating and managing the model's architecture.

### MATLAB
- **Computer Vision Toolbox**: For video capturing and processing.
- **Classification Learner**: For training models using the FER2013 dataset.

## Dataset

The system is trained on the **FER2013 dataset**, a popular dataset for facial expression recognition. It consists of grayscale 48x48 pixel images of faces categorized into 7 emotions:
1. **Angry**
2. **Disgust**
3. **Fear**
4. **Happy**
5. **Sad**
6. **Surprise**
7. **Neutral**

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).

## Installation

### Python Implementation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/real-time-emotion-recognition.git
   cd real-time-emotion-recognition
   ```

2. Download the FER2013 dataset and place it in the `data/` directory under `train/` and `test/` subdirectories.

### MATLAB Implementation

No specific installation is required for the MATLAB code. However, you should ensure that you have the following MATLAB toolboxes:
- Computer Vision Toolbox
- Image Processing Toolbox

## Usage

### Python

1. **Training the Model**:
   Run the Jupyter Notebook `emotion_recognition.ipynb` to train the model. The notebook preprocesses the FER2013 dataset, builds a CNN, and trains the model for emotion classification.

   ```bash
   jupyter notebook emotion_recognition.ipynb
   ```

2. **Real-Time Emotion Recognition**:
   Once the model is trained, you can use it to recognize emotions in real-time using your webcam.

   To run the real-time emotion recognition, execute the `real_time_emotion_recognition` function in the final cell of the notebook or from a Python script.

### MATLAB

1. Load the `FER2013` dataset into MATLAB.
2. Run the `emotion_recognition.m` script to process the images and start the real-time video emotion recognition (though the system is untested due to compilation time).

## Project Structure

```
real-time-emotion-recognition/
│
├── data/                       # Folder to place FER2013 dataset (train and test folders)
├── models/                     # Folder to save and load trained models
├── emotion_recognition.ipynb    # Python Jupyter Notebook
├── emotion_recognition.m        # MATLAB script
├── README.md                   # Project documentation
```

## Results

- **Training Accuracy**: The Python model was trained for **16 epochs** using the FER2013 dataset, achieving:
  - **Test Loss**: 1.2091
  - **Test Accuracy**: 54.88%
  
- **Real-Time Recognition**: The Python-based system successfully recognizes facial expressions from a live video feed using a webcam. The MATLAB version is written but untested due to compilation time.

## Future Improvements

Some improvements that could enhance the system include:
1. **Increase Training Epochs**: Training the model for more epochs to improve accuracy.
2. **Data Augmentation**: Applying techniques like flipping, rotation, and zoom to increase the diversity of the training data.
3. **Fine-Tuning the Model**: Experimenting with deeper neural networks or using transfer learning from pre-trained models (e.g., ResNet or VGG).
4. **Optimization in MATLAB**: Testing and optimizing the MATLAB implementation for performance.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit and push your changes (`git commit -am 'Add new feature'`).
5. Submit a pull request.
