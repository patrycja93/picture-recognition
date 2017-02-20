"""Given a picture returns its class to which it was assigned, and this picture."""
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import ORB

import sys

from image_clf import describe, histogramize, read

extractor = ORB(n_keypoints=200)
kmeans = read('kmeans')
classifier = read('classifier')
img = io.imread(sys.argv[1])

descriptor = describe(rgb2gray(img), extractor)
histogram = histogramize(descriptor, kmeans)
prediction = classifier.predict(histogram.reshape(1, -1))
print(prediction)

io.imshow(img)
io.show()
