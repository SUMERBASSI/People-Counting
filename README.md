Crowd Counting with CNN

Overview
This project implements a deep learning model for crowd counting using a convolutional neural network (CNN) based on the VGG16 architecture. The model predicts density maps from input images, allowing for accurate estimation of the number of people in crowded scenes.

Features
1. Gaussian Density Map Generation: Converts ground truth points into density maps using Gaussian distributions.
2. Data Augmentation: Includes preprocessing steps like normalization and resizing for robust training.
3. Model Training: Trains a CNN with custom layers for density map prediction, utilizing callbacks for improved performance.

Requirements
1. Python 3.x
2. TensorFlow
3. NumPy
4. Matplotlib
5. OpenCV
6. SciPy
7. Scikit-learn

Data Preparation
1. Input Images: Place images in the specified directory.
2. Ground Truth Maps: Store corresponding ground truth density maps in the designated folder.
   
Usage
1. Training: Run the training script to begin the training process. Adjust hyperparameters as needed (e.g., learning rate, batch size, number of epochs).
2. Testing: Test the model using new images to predict crowd density. Ensure the input image follows the expected format.

Results
The model outputs density maps indicating the predicted number of people in the input images. Comparison with ground truth data provides insight into accuracy.
