import numpy as np
from .metric import Metric

class ConfusionMatrix(Metric):
    """Constructs a confusion matrix for a multi-class classification problem.
    Does not support multi-label, multi-class problems.
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    """

    def __init__(self, num_classes, normalized=True):#False):
        super().__init__()
        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix
        The shape of the confusion matrix is K x K, where K is the number
        of classes.
        Keyword arguments:
        - predicted (numpy.ndarray): An N-tensor/array of integer values between 0 and K-1.
        - target (numpy.ndarray): An N-tensor/array of integer values between 0 and K-1.
        """
        print(target.max(),target.min(),self.num_classes)
        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
            'predicted values are not between 0 and k-1'

        assert (target.max() < self.num_classes) and (target.min() >= 0), \
            'target values are not between 0 and k-1'

        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def value(self):
        """Returns the confusion matrix."""
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf