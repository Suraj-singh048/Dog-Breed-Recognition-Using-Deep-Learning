Dog Breed Recognition Using Deep LearningProject OverviewThis project focuses on developing a deep learning model capable of accurately recognizing dog breeds from images. 
Utilizing Convolutional Neural Networks (CNNs), the model is trained on a dataset of dog images to classify the breed of a dog in a given image.
FeaturesHigh-Accuracy Dog Breed Classification: The model is designed to classify a wide range of dog breeds with high accuracy.
Image Preprocessing: Includes necessary preprocessing steps such as resizing, normalization, and data augmentation to improve model performance.
Top-5 Predictions: The model outputs the top 5 predicted dog breeds for a given image, ranked by confidence scores.
DatasetThe project utilizes a dataset comprising images of various dog breeds. 
For experimentation, two subsets of the dataset were used:
  Full Dataset: The entire dataset containing images of all the available dog breeds.
  Reduced Dataset: A subset containing 1000 images randomly sampled from the full dataset to compare training times and model performance.
Model ArchitectureThe project explores two primary model architectures:
  Custom CNN Model: A convolutional neural network designed from scratch, optimized for the specific task of dog breed classification.
  Pre-trained CNN Model: A transfer learning approach using a pre-trained model (such as ResNet50 or VGG16) fine-tuned for the dog breed classification task.
  TrainingCustom CNN Model: Trained on both the full dataset and the reduced dataset. 
Results are compared to understand the impact of dataset size on training time and model performance.
Pre-trained Model: Fine-tuned on the same datasets to leverage the benefits of transfer learning, potentially achieving higher accuracy with fewer training epochs.
EvaluationThe models are evaluated based on the following metrics:
  Accuracy: Overall accuracy of the model in predicting the correct dog breed.
  Top-5 Accuracy: Accuracy of the model in predicting the correct dog breed within the top 5 predictions.
  Confusion Matrix: A detailed confusion matrix to analyze the model's performance across different dog breeds.
Results: 
  The custom CNN model trained on the full dataset achieved an accuracy of 90%  
  The custom CNN model trained on the reduced dataset achieved an accuracy of 85%.
