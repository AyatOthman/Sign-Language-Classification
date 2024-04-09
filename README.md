# Sign-Language-Classification using Convolutional Neural Networks

This repository contains the code for building a Convolutional Neural Network (CNN) model to recognize hand gestures in American Sign Language (ASL). The model is trained on the Sign Language MNIST dataset, which consists of grayscale images representing individual letters from A to Z in ASL.

## Dataset
The Sign Language MNIST dataset used in this project contains two parts: training and testing data. Each image in the dataset is 28x28 pixels and represents a hand gesture corresponding to a specific letter of the alphabet.

- Training dataset: [sign_mnist_train.csv](/kaggle/input/sign-language-mnist/sign_mnist_train.csv)
- Testing dataset: [sign_mnist_test.csv](/kaggle/input/sign-language-mnist/sign_mnist_test.csv)

## Model Architecture
The CNN model architecture used for this task consists of multiple Conv2D layers followed by MaxPooling2D layers to extract features from the input images. The final layer is a Dense layer with softmax activation to classify the input into one of the 25 classes (letters A to Z).

## Training and Evaluation
The model is trained for 10 epochs using the Adam optimizer and categorical cross-entropy loss function. Training and validation accuracy and loss are plotted to analyze the model's performance. The trained model achieves an accuracy score of approximately 94.65% on the test dataset.

## Predictions
Random images from the test dataset are selected, and the model's predictions are compared with the true labels. The predicted labels along with the corresponding images are displayed to evaluate the model's performance visually.

Feel free to explore the code and experiment with different architectures or datasets!

