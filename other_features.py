# USAGE
# python other_features.py --dataset covid_images

# import the necessary packages
from skimage.filters import prewitt_h,prewitt_v
from skimage.feature import greycomatrix, greycoprops
from sklearn.decomposition import PCA
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

	pca = PCA(n_components = 2)
	# pixels = pixels.reshape(-1, 1)
	reduced_image = pca.fit_transform(pixels)
	# new_image = reduced_image.astype(np.integer)
	# new_image[new_image < 0] = 0
	# comatrix = greycomatrix(new_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=4)
	rawImages.append(pixels) # update the raw images and labels matrices,
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

# testRI is the raw pixels of each of the images and testRL are the classes
# print("testRI is ", testRI)
# print("testRL is ", testRL)

# Plotting decision boundary

# h = .02  # step size in the mesh

# # # Create color maps
# cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
# cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

# # # Plot the decision boundary. For that, we will assign a color to each
# # # point in the mesh [x_min, x_max]x[y_min, y_max].
# x_min, x_max = trainRI[:, 0].min() - 1, trainRI[:, 0].max() + 1
# y_min, y_max = trainRI[:, 1].min() - 1, trainRI[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
# print("what we're predicting", np.c_[xx.ravel(), yy.ravel()])
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# # # Plot also the training points - TrainRL needs to be ints
# plt.scatter(trainRI[:, 0], trainRI[:, 1], c=trainRL, cmap=cmap_bold,
#             edgecolor='k', s=20)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("3-Class classification (k = %i, weights = '%s')" % (1, "distance"))

# plt.show()

# Plotting decision thing



# for i, item in enumerate(trainRL):
# 	if item == 'noncovid':
# 		trainRL[i] = 0
# 	else:
# 		trainRL[i] = 1

# print("trainRI is", trainRI)

# y = trainRL.astype(np.integer)
# # plot_decision_regions(trainRI, y, clf=model, legend=2)

# pca = PCA(n_components = 2)
# X_train2 = pca.fit_transform(trainRI)
# model.fit(X_train2, y)
# plot_decision_regions(X_train2, y, clf=model, legend=2)

# # Adding axes annotations
# # plt.xlabel("X")
# # plt.ylabel("Y")
# plt.title("K nearest neighbor algorithm with PCA & K=1")
# plt.show()

# https://towardsdatascience.com/knn-visualization-in-just-13-lines-of-code-32820d72c6b6