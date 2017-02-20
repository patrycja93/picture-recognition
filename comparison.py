"""
Compare two classifiers, svc, rtf.

Pay result of the comparison.
"""

from image_clf import describe, histogramize, read
from skimage import color, io, feature
import numpy as np
import os

def main(path):
    """
    Compare svc and rtf classifiers.

    With the directory specified by the user, main function is no longer collects 100 images.
    Then looks for files with the extension .pickle, which are stored models for previously trained directories.
    Each photo in a user directory, reduced to shades of gray, pulls marks and creates histograms.
    If any of the pictures has been classified in another class than it should function prints a number of errors and the class name.
    """
    _, cls = os.path.split(path)
    des = feature.ORB(n_keypoints=200)
    rtf = read('rtf')
    clu_rtf = read('rtf_kmeans')
    svc = read('svc')
    clu_svc = read('svc_kmeans')

    filenames = os.listdir(path)
    np.random.shuffle(filenames)
    filenames = filenames[:100]
    images = [io.imread(os.path.join(path, filename)) for filename in filenames]
    images = [color.rgb2gray(img) for img in images]

    features = [describe(img, des) for img in images]

    print("rtf")
    histograms = [histogramize(feat, clu_rtf) for feat in features]
    rtf_result = rtf.predict(histograms)

    print('svc')
    histograms = [histogramize(feat, clu_svc) for feat in features]
    svc_result = svc.predict(histograms)

    rtf_faults = rtf_result[rtf_result!=cls]
    svc_faults = svc_result[svc_result!=cls]

    print("RTF faults (", rtf_faults.size, "):", rtf_faults)
    print("SVC faults (", svc_faults.size, "):", svc_faults)

if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    main(path)
