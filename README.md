# 🐶 Dog Breed Recognition Using Deep Learning

![Dog Breed Recognition](https://img.icons8.com/color/96/000000/dog.png)

## Project Overview

This project focuses on developing a deep learning model capable of accurately recognizing dog breeds from images. Utilizing Convolutional Neural Networks (CNNs), the model is trained on a comprehensive dataset of dog images to classify the breed of a dog in a given image. The project culminates in a user-friendly **Streamlit** application deployed on [Streamlit Cloud](https://streamlit.io/cloud), allowing users to interactively identify dog breeds with ease.

## 🚀 Live Demo

Experience the Dog Breed Identification App live!

👉 **[Try the App Here](https://recognize-your-dog-by-sps.streamlit.app/)**


## 🛠 Features

- **High-Accuracy Dog Breed Classification**: The model classifies a wide range of dog breeds with high precision.
- **Image Preprocessing**: Implements resizing, normalization, and data augmentation to enhance model performance.
- **Top-5 Predictions**: Displays the top 5 predicted dog breeds for a given image, ranked by confidence scores.
- **User-Friendly Interface**: Upload an image or capture a photo directly from your device's camera to get instant predictions.
- **Responsive Design**: Accessible across various devices including desktops, tablets, and smartphones.
- **Minimalistic and Attractive UI**: Clean and intuitive design for a seamless user experience.

## 📁 Dataset

The project utilizes a dataset comprising images of various dog breeds. For experimentation, two subsets of the dataset were used:

- **Full Dataset**: The entire dataset containing images of all available dog breeds.
- **Reduced Dataset**: A subset containing 1,000 images randomly sampled from the full dataset to compare training times and model performance.

*Dataset Source: [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) (Ensure to comply with licensing and usage terms)*

## 🧠 Model Architecture

The project explores two primary model architectures:

1. **Custom CNN Model**: A convolutional neural network designed from scratch, optimized for the specific task of dog breed classification.
2. **Pre-trained CNN Model**: A transfer learning approach using a pre-trained model (such as ResNet50 or VGG16) fine-tuned for the dog breed classification task.

## 🎯 Training

### Custom CNN Model

- **Datasets**: Trained on both the full dataset and the reduced dataset.
- **Objective**: Compare the impact of dataset size on training time and model performance.

### Pre-trained Model

- **Approach**: Fine-tuned on the same datasets to leverage the benefits of transfer learning.
- **Advantage**: Achieves higher accuracy with fewer training epochs compared to training from scratch.

## 📈 Evaluation

The models are evaluated based on the following metrics:

- **Accuracy**: Overall accuracy of the model in predicting the correct dog breed.
- **Top-5 Accuracy**: Accuracy of the model in predicting the correct dog breed within the top 5 predictions.
- **Confusion Matrix**: A detailed confusion matrix to analyze the model's performance across different dog breeds.

### Results

- **Custom CNN Model**:
  - **Full Dataset**: Achieved an accuracy of **90%**.
  - **Reduced Dataset**: Achieved an accuracy of **85%**.
