# human-emotion-detection
This project focuses on emotion recognition from facial expressions using Convolutional Neural Networks (CNNs). The model was trained on the FER-2013 dataset, which consists of a wide variety of images labeled with one of seven emotions: Anger, Disgust, Fear, Happiness, Neutral, Sadness, and Surprise.

Initially, the model achieved an accuracy of approximately 50%. After applying various optimizations, including data augmentation, dropout layers, and fine-tuning the model architecture, the accuracy improved to 55% on the test set. While this is a modest improvement, it demonstrates the power of CNNs for image classification tasks in a real-world scenario.

The key features of the project include:

Data Augmentation: Used to artificially increase the diversity of the training dataset by applying transformations such as rotations, zoom, shifts, and flips.
CNN Architecture: A deep neural network with several convolutional layers designed to extract spatial features from images.
Dropout Layers: Used to prevent overfitting by randomly disabling a fraction of the neurons during training.
Model Evaluation: The model's performance was evaluated on a separate test set to check the generalization ability.
The model was trained and tested in Google Colab, leveraging the power of GPUs for faster computation. The dataset was downloaded directly from Kaggle, ensuring a seamless integration with the training pipeline.

In the future, the model could be further improved by:

Exploring advanced CNN architectures (e.g., ResNet, InceptionNet).
Fine-tuning hyperparameters for optimal performance.
Using a larger and more diverse dataset to capture a wider range of emotional expressions.
