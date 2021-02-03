# USAGE - python knn_classifier.py --dataset covid_images

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import cv2
import os
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
	labels.append(label)

	# show an update every 100 images
	if i > 0 and i % 100 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

rawImages = np.array(rawImages)
labels = np.array(labels)

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

# Plotting decision thing
for i, item in enumerate(trainRL):
	if item == 'noncovid':
		trainRL[i] = 0
	else:
		trainRL[i] = 1

y = trainRL.astype(np.integer)

pca = PCA(n_components = 2)
X_train2 = pca.fit_transform(trainRI)
model.fit(X_train2, y)
plot_decision_regions(X_train2, y, clf=model, legend=2)

plt.title("K nearest neighbor algorithm with PCA & K=1")
plt.show()