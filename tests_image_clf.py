"""Module with tests."""
from skimage import feature
from sklearn import ensemble
import numpy as np

import os
from image_clf import Pipeline
import pytest


@pytest.fixture
def pipeline():
    """Check have been selected marks."""
    return Pipeline(
            feature.ORB(n_keypoints=200),
            ensemble.RandomForestClassifier(),
    )


@pytest.fixture
def mock_imgpaths():
    """Check whether the first image from the first directory was downloaded."""
    return [os.path.join(
            #'101_ObjectCategories',
            #'accordion',
            'CollectionsOfPhotos',
            'accordion',
            'image_0001.jpg',
    )]


@pytest.fixture
def mock_labels(mock_imgpaths):
    """Check if the selected photos from the catalog."""
    return ['cls'] * len(mock_imgpaths)


@pytest.fixture
def mock_feats():
    """Check if create array."""
    return np.arange(200*256).reshape((200, -1))


def test_load_files(pipeline, mock_imgpaths):
    """Check if the files have been load."""
    assert not pipeline._images
    pipeline._load_files(mock_imgpaths)
    assert pipeline._images


def test_extract_features(pipeline, mock_imgpaths):
    """Check if the features have been extract."""
    pipeline._load_files(mock_imgpaths)

    assert not pipeline._features
    pipeline._extract_features()
    assert pipeline._features

def test_clusterize_features(pipeline, mock_imgpaths, mock_feats):
    """Check if the features have been clusterize."""
    pipeline._load_files(mock_imgpaths)
    pipeline._extract_features()

    assert not hasattr(pipeline._clusterizer, 'labels_')
    pipeline._clusterize_features(mock_feats)
    assert hasattr(pipeline._clusterizer, 'labels_')

def test_histogramize(pipeline, mock_imgpaths, mock_feats):
    """Check if the histograms have been created."""
    pipeline._load_files(mock_imgpaths)
    pipeline._extract_features()
    pipeline._clusterize_features(mock_feats)

    assert not pipeline._histograms
    pipeline._histogramize()
    assert pipeline._histograms

def test_classifier(pipeline, mock_imgpaths, mock_feats, mock_labels):
    """Check action of classifiers."""
    pipeline._load_files(mock_imgpaths)
    pipeline._extract_features()
    pipeline._clusterize_features(mock_feats)
    pipeline._histogramize()

    assert not hasattr(pipeline._clf, 'classes_')
    pipeline._train_model(mock_labels)
    assert hasattr(pipeline._clf, 'classes_')
