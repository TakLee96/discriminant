import numpy as np
from os import path
from pickle import load, dump
from scipy.io import loadmat, savemat
from classifier import classify


raw = loadmat(path.join(path.dirname(__file__), "features.mat"))
raw_data = raw['data']
raw_label = raw['label'][0]
np.random.seed(0)
ordering = np.random.permutation(len(raw_data))
data = np.ndarray(shape=raw_data.shape, dtype=raw_data.dtype)
label = np.ndarray(shape=raw_label.shape, dtype=raw_label.dtype)
for old, new in enumerate(ordering):
    data[new] = raw_data[old]
    label[new] = raw_label[old]


def validation():
    holdout_data = data[-2000:]
    holdout_label = label[-2000:]
    training_data = data[:-2000]
    training_label = label[:-2000]
    classifier = classify(training_data, training_label, which="lda")
    score = classifier.score(classifier.classify_all(training_data), training_label)
    print "training with", len(training_data), "data get prediction rate", score
    score = classifier.score(classifier.classify_all(holdout_data), holdout_label)
    print "    and holdout rate", score
# validation()
