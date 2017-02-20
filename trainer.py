"""
The trainer creates models.

If the user gives the name of the directory with pictures without indicating a specific image to recognize this function creates three directories.
Next groups images from the directory specified according to their classification.
"""
from skimage import io
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import ORB

import numpy as np
import os
import sys
import shutil
from image_clf import describe, histogramize, persist


path = sys.argv[1]

filenames = [name for name in os.listdir(path)]
isdir = lambda filename: os.path.isdir(os.path.join(path, filename))
directories = list(filter(isdir, filenames))
directories = sorted(directories, key=lambda s: s.lower())
isfile = lambda filename: os.path.isfile(os.path.join(path, filename))
directories = directories[:3]

print('dirs:', directories)


classes = []
if directories:
    images = []
    for cls in directories:
        filenames = os.listdir(os.path.join(path, cls))[:10]
        filenames = list(map(lambda x: os.path.join(path, cls, x), filenames))
        classes += [cls] * len(filenames)
        images += filenames
else:
    filenames = list(filter(isfile, filenames))
    filenames = [os.path.join(path, filename) for filename in filenames]
    images = filenames


print("Reading", len(images), "images.")
images = [io.imread(img) for img in images]


print("Converting images to grayscale.")
images = [rgb2gray(img) for img in images]


extractor = ORB(n_keypoints=200)

print("Extracting features from images.")
descriptors = [describe(img, extractor) for img in images]

print("Clusterizing features..", end='')
kmeans = KMeans()
# descriptors_sample  = np.random.choice(descriptors, len(descriptors))
kmeans.fit(np.vstack(descriptors))
print(" done")


histograms = [histogramize(des, kmeans) for des in descriptors]

if classes:
    classifier = RandomForestClassifier()
    # classifier = SVC()
    classifier.fit(histograms, classes)
else:
    classifier = KMeans(n_clusters=3)
    classifier.fit(histograms)
    classes = classifier.predict(histograms)
    filenames = np.array(filenames)
    for i in range(classifier.n_clusters):
        paths = filenames[classes==i]
        dirname = 'class{}'.format(i)

        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.mkdir(dirname)
        for path in paths:
            _, filename = os.path.split(path)
            new_path = os.path.join(dirname, filename)
            shutil.copyfile(path, new_path)


persist(classifier, 'classifier')
persist(kmeans, 'kmeans')
