import numpy as np
import numpy.linalg as la


def gaussian_estimate(data, label):
    assert len(data) == len(label), "label length mismatch"
    assert label.min() == 0, "label should start from 0"
    assert label.max() != 0, "label should have multiple"
    classified = [list() for _ in range(label.max() + 1)]
    for i in range(len(label)):
        classified[label[i]].append(data[i])
    for i in range(len(classified)):
        classified[i] = np.array(classified[i])
    log_priors = np.array([np.log(1.0 * len(classified[i]) / len(label)) for i in range(len(classified))])
    means = np.array([np.mean(points, axis=0) for points in classified])
    variances = np.array([np.cov(np.transpose(points), ddof=0) for points in classified])
    return log_priors, means, variances


class Classifier:
    def __init__(self, data, label):
        self.log_priors, self.means, self.variances = gaussian_estimate(data, label)
        self.num_classes = len(self.log_priors)

    def classify(self, point):
        raise Exception("unimplemented")

    def classify_all(self, points):
        return np.apply_along_axis(lambda p: self.classify(p), 1, points)

    def score(self, predictions, label):
        return 1.0 * np.sum(np.equal(predictions, label)) / len(label)


class LDAClassifier(Classifier):
    def __init__(self, data, label):
        Classifier.__init__(self, data, label)
        self.precision = la.pinv(np.sum(self.variances, axis=0))
        self.variances = None

    def classify(self, point):
        discriminant = np.ndarray(shape=(self.num_classes,), dtype=float)
        for i in range(self.num_classes):
            discriminant[i] = np.dot(np.dot(self.means[i], self.precision),
                                     point - np.divide(self.means[i], 2)) + self.log_priors[i]
        return np.argmax(discriminant)


class QDAClassifier(Classifier):
    def __init__(self, data, label):
        Classifier.__init__(self, data, label)
        self.precisions = np.array([la.pinv(self.variances[i]) for i in range(self.num_classes)])
        self.log_determinants = np.array([la.slogdet(self.variances[i])[1] for i in range(self.num_classes)])
        self.variances = None

    def classify(self, point):
        discriminant = np.ndarray(shape=(self.num_classes,), dtype=float)
        for i in range(self.num_classes):
            normalized = point - self.means[i]
            discriminant[i] = self.log_priors[i] - self.log_determinants[i] / 2 \
                              - np.dot(np.dot(normalized, self.precisions[i]), normalized) / 2
        return np.argmax(discriminant)


def classify(data, label, which):
    if which == "lda":
        return LDAClassifier(data, label)
    elif which == "qda":
        return QDAClassifier(data, label)
    raise Exception("lda or qda")


__all__ = [gaussian_estimate, classify]
