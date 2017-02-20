"""
Create models.

Creates models folders from a specified directory.
These models saves files with the extension .pickle.
"""
from skimage import feature
from sklearn import cluster, svm
from image_clf import Pipeline, persist

def main(path, n_imgs_per_class=100):
    """
    Get the first three folders with photos from the folder.

    Creates for each of these models for a particular classifier.
    """
    import os
    filenames = [name for name in os.listdir(path)]
    isdir = lambda filename: os.path.isdir(os.path.join(path, filename))
    dirs = list(filter(isdir, filenames))
    dirs = sorted(dirs, key=lambda s: s.lower())[:3]
    dirs = dirs[:3]

    filepaths = []
    labels = []
    if dirs:  # supervised learning
        for cls in dirs:
            filenames = os.listdir(os.path.join(path, cls))
            filenames = sorted(filenames)
            filenames = filenames[:n_imgs_per_class]
            filenames = list(map(lambda x: os.path.join(path, cls, x), filenames))
            labels += [cls] * len(filenames)
            filepaths += filenames


    print("Classes: ", *dirs)
    for f in filepaths:
        print(f)

    for fp, l in zip(filepaths, labels):
        print(l, fp)


    if labels:
    #    clf = ensemble.RandomForestClassifier(n_estimators=50)
        clf = svm.SVC()
    else:
        clf = cluster.KMeans()

    pipeline = Pipeline(
            feature.ORB(n_keypoints=200),
            clf
    )
    pipeline.train_model(filepaths, labels)

    print("Persisting model..")
    persist(pipeline._clf, 'classifier')
    persist(pipeline._clusterizer, 'kmeans')


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    main(path)
