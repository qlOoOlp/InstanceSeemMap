# import numpy as np
# from .metric import Metric
# from .confusionmatrix import ConfusionMatrix

# class IoU(Metric):
#     """Computes the intersection over union (IoU) per class and corresponding
#     mean (mIoU).
#     Intersection over union (IoU) is a common evaluation metric for semantic
#     segmentation.
#     """

#     def __init__(self, num_classes, normalized=False, ignore_index=None):
#         super().__init__()
#         self.conf_metric = ConfusionMatrix(num_classes, normalized)
#         if ignore_index is None:
#             self.ignore_index = None
#         elif isinstance(ignore_index, int):
#             self.ignore_index = (ignore_index,)
#         else:
#             self.ignore_index = tuple(ignore_index)

#     def reset(self):
#         self.conf_metric.reset()

#     def add(self, predicted, target):
#         """Adds the predicted and target pair to the IoU metric."""
#         self.conf_metric.add(predicted.flatten(), target.flatten())

#     def value(self):
#         """Computes the IoU and mean IoU."""
#         conf_matrix = self.conf_metric.value()
#         if self.ignore_index is not None:
#             for index in self.ignore_index:
#                 conf_matrix[:, index] = 0
#                 conf_matrix[index, :] = 0

#         true_positive = np.diag(conf_matrix)
#         false_positive = np.sum(conf_matrix, axis=0) - true_positive
#         false_negative = np.sum(conf_matrix, axis=1) - true_positive

#         total_pixels = np.sum(conf_matrix)

#         acc = np.sum(true_positive) / np.sum(conf_matrix)

#         with np.errstate(divide='ignore', invalid='ignore'):
#             iou = true_positive / (true_positive + false_positive + false_negative)

#         if self.ignore_index is not None:
#             for index in self.ignore_index:
#                 iou[index] = np.nan

#         miou = np.nanmean(iou)

#         with np.errstate(divide='ignore', invalid='ignore'):
#             recalls = true_positive / np.sum(conf_matrix, 1)

#         if self.ignore_index is not None:
#             for index in self.ignore_index:
#                 recalls[index] = np.nan

#         recall = np.nanmean(recalls)

#         with np.errstate(divide='ignore', invalid='ignore'):
#             precisions = true_positive / np.sum(conf_matrix, 0)

#         if self.ignore_index is not None:
#             for index in self.ignore_index:
#                 precisions[index] = np.nan

#         precision = np.nanmean(precisions)

#         return iou, miou, acc, recalls, recall, precisions, precision



import numpy as np
from .metric import Metric
from .confusionmatrix import ConfusionMatrix

class IoU(Metric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU), Frequency Weighted IoU (FWIoU), mean F1 score, mean Recall, and mean Precision.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)
        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            self.ignore_index = tuple(ignore_index)

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric."""
        self.conf_metric.add(predicted.flatten(), target.flatten())

    def value(self):
        """Computes the IoU, mean IoU, FWIoU, mean Recall, mean Precision, and mean F1 score."""
        conf_matrix = self.conf_metric.value()
        print(type(conf_matrix))
        print(conf_matrix.shape)
        # conf_matrix[0,:]=0
        # conf_matrix[1,:]=0
        # conf_matrix[2,:]=0
        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf_matrix[:, index] = 0
                conf_matrix[index, :] = 0

        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, axis=0) - true_positive
        false_negative = np.sum(conf_matrix, axis=1) - true_positive
        total_pixels = np.sum(conf_matrix)

        acc = np.sum(true_positive) / total_pixels

        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        if self.ignore_index is not None:
            for index in self.ignore_index:
                iou[index] = np.nan

        miou = np.nanmean(iou)

        # FWIoU Calculation
        freq = np.sum(conf_matrix, axis=1) / total_pixels
        fwiou = np.nansum(freq * iou)


        # Precision and Recall Calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            precisions = true_positive / (true_positive + false_positive)
            recalls = true_positive / (true_positive + false_negative)

        if self.ignore_index is not None:
            for index in self.ignore_index:
                precisions[index] = np.nan
                recalls[index] = np.nan

        precision = np.nanmean(precisions)
        recall = np.nanmean(recalls)

        # F1 Score Calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

        if self.ignore_index is not None:
            for index in self.ignore_index:
                f1_scores[index] = np.nan

        mean_f1 = np.nanmean(f1_scores)

        return iou, miou, acc, fwiou, mean_f1, recalls, recall, precisions, precision