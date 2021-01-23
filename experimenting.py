# USAGE
# python experimenting.py --dataset covid_images

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# np.set_printoptions(threshold=np.inf)

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
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
# imagePaths is the list of all images in my folder
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the raw pixel intensities matrix and labels list
rawImages = []
labels = []

# Try this for 1 input image

# load the image and extract the class label
imagePath = 'covid_images/noncovid.105.jpg'
image = cv2.imread(imagePath)
print("image is", image)

# counter=0
# level=0
# shout=0

# for i in image:
#     level+=1
#     for x in i:
#         counter+=1
#         for y in x:
#             shout+=1
# # print("size of image is", len(image)) # 3d list, list inside of list inside of list. Outer level is 137
# print("size of outer image is", level)
# print("size of inner image is", counter)
# print("size of innermost image is", shout)

label = imagePath.split(os.path.sep)[-1].split(".")[0]
print("The label is", label)

# extract raw pixel intensity features
pixels = image_to_feature_vector(image)
print("pixel length is: ", len(pixels)) # A color 1D array of 32x32x3 = 3079

# update the raw images and labels matrices,
rawImages.append(pixels)
labels.append(label)

# show an update every 100 images
# if i > 0 and i % 100 == 0:
#     print("[INFO] processed {}/{}".format(i, len(imagePaths)))

rawImages = np.array(rawImages) # array of the pixels
labels = np.array(labels) # array of the labels

# partition the data into training and testing splits, using 80%
# of the data for training and the remaining 20% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.2, random_state=42)

# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"], metric='manhattan')
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
# predictions = model.kneighbors_graph(testRI)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# testRI is the raw pixels of each of the images and testRL are the classes
print("testRI is ", testRI)
print("testRL is ", testRL)