import numpy as np

def mAP_func(y_true, y_pred):
    num_classes = y_true.shape[1]
    average_precisions = []
    print('true', y_true.shape)
    print('pred', y_pred.shape)

    for index in range(num_classes):
        pred = y_pred[:,index]
        label = y_true[:,index]


        sorted_indices = np.argsort(-pred)
        sorted_pred = pred[sorted_indices]
        sorted_label = label[sorted_indices]


        tp = (sorted_label == 1)
        fp = (sorted_label == 0)

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        npos = np.sum(sorted_label)

        recall = tp * 1.0 / (npos+1e-5)

        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp*1.0 / np.maximum((tp + fp), np.finfo(np.float64).eps)

        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        average_precisions.append(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    # print('average precision', np.mean(average_precisions, dtype=np.float32))
    return np.mean(average_precisions, dtype=np.float32)