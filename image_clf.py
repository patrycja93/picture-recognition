"""
This module contains core functions used in image classification.
"""

from skimage import color, io
from sklearn import cluster

import pickle
import numpy as np


def persist(obj, name):
    """Persists object in file with given name."""
    with open(name + '.pickle', 'wb') as fh:
        pickle.dump(obj, fh)


def read(name):
    """Reads object from file with given name."""
    with open(name + '.pickle', 'rb') as fh:
        return pickle.load(fh)


def describe(image, descriptor):
    descriptor.detect_and_extract(image)
    return descriptor.descriptors


def histogramize(features, clusterizer) -> np.array:
    histogram, _ = np.histogram(clusterizer.predict(features))
    return histogram


class Pipeline:
    """
    """

    def __init__(self, des, clf):
        self._clf = clf
        self._des = des
        self._clusterizer = cluster.KMeans()
        self._images = []
        self._features = []
        self._histograms = []

    def _load_files(self, filenames):
        """"""
        images = [io.imread(fn) for fn in filenames]
        images = [color.rgb2gray(img) for img in images]
        self._images = images

    def _extract_features(self):
        self._features = [describe(img, self._des) for img in self._images]

    def _clusterize_features(self, features):
        self._clusterizer.fit(features)

    def _histogramize(self):
        self._histograms = [
                histogramize(ft, self._clusterizer) for ft in self._features
        ]

    def _train_model(self, labels=None):
        if labels:
            self._clf.fit(self._histograms, labels)
        else:
            self._clf.fit(self._histograms)

    def train_model(self, filenames, labels=None):
        print("Loading", len(filenames), "files..")
        self._load_files(filenames)
        assert self._images
        print("Extracting features..")
        self._extract_features()
        assert self._features
        print("Clustering features..")
        self._clusterize_features(np.vstack(self._features))
        print("Creating feature histograms..")
        self._histogramize()
        assert self._histograms
        print("Training classifier..")
        self._train_model(labels)

    def predict(self, filename):
        img = io.imread(filename)
        img = color.rgb2gray(img)
        feat = describe(img, self._des)
        hist = histogramize(feat, self._clusterizer)
        return self._clf.predict(hist)
