# USAGE - python nn_classifier.py --dataset covid_images

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the raw pixel intensities matrix and labels list
rawImages = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):

	# load the image and extract the class label
	image = cv2.imread(imagePath)	
	label = imagePath.split(os.path.sep)[-1].split(".")[0]

	# extract raw pixel intensity features
	pixels = image_to_feature_vector(image)

	# update the raw images and labels matrices,
	rawImages.append(pixels)
	if label == 'covid':
		labels.append(1)
	elif label == 'noncovid':
		labels.append(0)

	# show an update every 100 images
	if i > 0 and i % 100 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

rawImages = np.array(rawImages)
labels = np.array(labels)

# Neural net
model = keras.Sequential()
model.add(Dense(18, activation=tf.nn.relu))
model.add(Dense(2, activation=tf.nn.softmax))  # Puts results in array of 2 probabilities

# # Conv neural net
# rawImages = rawImages.reshape(-1, 746, 3072)
# model = Sequential()
# model.add(Conv1D(256, (1)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=(1)))
# model.add(Conv1D(256, (1)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=(1)))
# model.add(Dense(7))
# model.add(Activation('sigmoid'))

# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# accFeats is Data (2d matrix), actID is labels (class #)
model.fit(rawImages, labels, batch_size=1, epochs=10)

test_loss, test_acc = model.evaluate(rawImages, labels)  # gets test accuracies
print('Test acc:', test_acc)