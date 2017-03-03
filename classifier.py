import numpy as np
import numpy.linalg as la


def gaussian_estimate(data, label):
    assert len(data) == len(label), "label length mismatch"
    assert label.min() == 0, "label should start from 0"
    assert label.max() != 0, "label should have multiple"
    trim = np.sum(data, axis=0) > 0
    data = data[:, trim]
    classified = [list() for _ in range(label.max() + 1)]
    for i in range(len(label)):
        classified[label[i]].append(data[i])
    for i in range(len(classified)):
        classified[i] = np.array(classified[i])
    log_priors = np.array([np.log(1.0 * len(classified[i]) / len(label)) for i in range(len(classified))])
    means = np.array([np.mean(points, axis=0) for points in classified])
    variances = np.array([np.cov(np.transpose(points)) for points in classified])
    return log_priors, means, variances, trim


class GaussianClassifier:
    def __init__(self, data, label):
        self.log_priors, self.means, self.variances, self.trim = gaussian_estimate(data, label)
        self.num_classes = len(self.log_priors)
        self.m, self.d = data.shape

    def classify(self, point):
        raise Exception("unimplemented")

    def classify_all(self, points):
        return np.apply_along_axis(lambda p: self.classify(p), 1, points)

    @staticmethod
    def score(predictions, label):
        return 1.0 * np.sum(np.equal(predictions, label)) / len(label)


class LDAClassifier(GaussianClassifier):

    def __init__(self, data, label, alpha=1e-6):
        GaussianClassifier.__init__(self, data, label)
        self.variance = np.sum(self.variances, axis=0) + alpha * np.eye(self.d)
        del self.variances
        self.precisions = np.ndarray(shape=(self.num_classes, self.d), dtype=self.variance.dtype)
        for i in range(self.num_classes):
            self.precisions[i] = la.solve(self.variance, self.means[i])
        del self.variance

    def classify(self, point):
        point = point[self.trim]
        discriminant = np.ndarray(shape=self.num_classes, dtype=np.float)
        for i in range(self.num_classes):
            discriminant[i] = np.dot(point - np.divide(self.means[i], 2),
                                     self.precisions[i]) + self.log_priors[i]
        return np.argmax(discriminant)


class QDAClassifier(GaussianClassifier):

    def __init__(self, data, label):
        GaussianClassifier.__init__(self, data, label)
        self.log_determinants = np.ndarray(shape=self.num_classes, dtype=np.float)
        for i in range(self.num_classes):
            s, logdet = la.slogdet(self.variances[i])
            if s == 0:
                raise Exception("singular matrix")
            self.log_determinants[i] = logdet
        self.precisions = np.array([la.inv(self.variances[i]) for i in range(self.num_classes)])
        del self.variances

    def classify(self, point):
        discriminant = np.ndarray(shape=(self.num_classes,), dtype=float)
        for i in range(self.num_classes):
            normalized = point - self.means[i]
            discriminant[i] = self.log_priors[i] - self.log_determinants[i] / 2 \
                              - np.dot(np.dot(normalized, self.precisions[i]), normalized) / 2
        return np.argmax(discriminant)


__all__ = [gaussian_estimate, LDAClassifier, QDAClassifier]
