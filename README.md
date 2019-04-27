# Handwriting-Recognition

In this project I implemented a handwriting detector using HOG. HOG is basically a feature descriptor thorough which the local object appearance and shape within an image can be described by the distribution of intensity gradients or edge functions.
 
In this project, 4 python scripts were written. The first one is :

Hog.py:
In this, we use the feature sub-package of scikit-image library. This feature package contains many methods to extract features from images. We use the hog method. Then we create a class named HOG that uses this hog feature extraction method. This class computes the histogram of gradient magnitudes for each cell in an image. Then we define a method named describe that takes the image as the input argument and finally returns the resulting HOG feature vector. 


The second is:
Dataset.py:
For training our model, we need a dataset. The dataset used is the MNIST digit recognition dataset. Before training our model, we first need to augment the dataset, that is, data manipulation is done so as to train the model well. For this purpose, imutils is used that performs operations such as resizing, rotation of the images. After this, the deskewing of the digits is done because everyone has a different writing style with different skew angles. We bring the digits to a standard skew angle by deskewing them.  


The third is:
Train.py:
For training the model, we need the LinearSVC model from scikit-learn library to train a linear support vector machine. We also need the HOG descriptor and dataset utility functions that we created. The first step is to load the dataset. We then initialize the HOG descriptor. The HOG feature vector is then computed for the pre-processed image by using the describe method that we created in the HOG class. Now after initializing the LinearSVC, the model is trained using the dataset. Finally, the trained model is saved by using joblib. 


The fourth is:
Classify.py:
Now this trained model is used to classify the digits in images. Just as in the training phase, we requires the usage of HOG for image descriptor and dataset for pre-processing utilities. First we load the trained model and the image from which the digits are to be recognized and the HOG descriptor is initialized. The input image is then blurred and subjected to canny edge detector to find the edges in the image. after this, we find the contours in the edged image. Each of the contours represent a digit in the image that needs to be classified.  After this we create a rectangular bounding box for each contour. These bounding boxes are now our ROI. Finally the digit is classified by first computing the HOG feature vector on our ROI, then this HOG feature vector is fed into the LinearSVC’s  predict method which classifies which digit the ROI is. 
