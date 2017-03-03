import numpy as np
from os import path
from scipy.io import loadmat
from timer import timer
from classifier import LDAClassifier, QDAClassifier


""" TODO: choose either mnist or spam >>HERE<< """
which = "spam.mat"
which = "mnist.mat"
""" TODO: choose either mnist or spam >>HERE<< """


timer.start("reading", which, "data from matlab file")
raw = loadmat(path.join(path.dirname(__file__), "data", which))
raw_data = raw['data']
raw_labl = raw['label'][0]
timer.end("done")

timer.start("permuting data randomly")
np.random.seed(0)
ordering = np.random.permutation(len(raw_data))
data = np.ndarray(shape=raw_data.shape, dtype=raw_data.dtype)
labl = np.ndarray(shape=raw_labl.shape, dtype=raw_labl.dtype)
for old, new in enumerate(ordering):
    data[new] = raw_data[old]
    labl[new] = raw_labl[old]
del raw, raw_data, raw_labl, ordering
timer.end("done")


def cross_validation(method, k=5):
    if method == "lda":
        Classifier = LDAClassifier
    elif method == "qda":
        Classifier = QDAClassifier
    else:
        raise Exception("lda or qda only")

    timer.start("folding data into", k, "copies")
    data_slice = [ None ] * k
    labl_slice = [ None ] * k
    train_rate = [ 0.0 ] * k
    valid_rate = [ 0.0 ] * k
    n = len(labl)
    m = n / k
    for i in range(k):
        data_slice[i] = data[(i*m):min((i+1)*m,n)]
        labl_slice[i] = labl[(i*m):min((i+1)*m,n)]
    timer.end("done")

    for j in range(k):
        timer.start("validation iteration #", j)
        training_data = np.concatenate(tuple(data_slice[i] for i in range(k) if i != j))
        training_labl = np.concatenate(tuple(labl_slice[i] for i in range(k) if i != j))
        print ".... data formating done"
        c = LDAClassifier(training_data, training_labl)
        print ".... classifier training done"
        train_rate[j] = c.score(c.classify_all(training_data), training_labl)
        print ".... training accuracy computation done"
        valid_rate[j] = c.score(c.classify_all(data_slice[j]), labl_slice[j])
        print ".... validation accuracy computation done"
        timer.end("done; training accuracy =", train_rate[j], "; validation accuracy =", valid_rate[j])

    print k, "fold cross validation for", method, "on dataset", which, "complete"
    print ".... overall training accuracy   =", np.mean(train_rate)
    print ".... overall validation accuracy =", np.mean(valid_rate)


cross_validation("qda")
