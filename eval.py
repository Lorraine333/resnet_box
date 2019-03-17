import numpy as np

def mAP_func(y_true, y_pred):
    num_classes = y_true.shape[1]
    average_precisions = []
    print('true', y_true)
    print('pred', y_pred)

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

def best_accuracy(y_pred, target):
    y_pred = np.reshape(y_pred, (-1))
    target = np.reshape(target, (-1))
    # print(y_pred, target)
    errs = -y_pred
    # best_threshold choosing by maximizing accuracy
    indices = np.argsort(errs)
    sortedErrors = errs[indices]
    sortedTarget = target[indices]
    tp = np.cumsum(sortedTarget)
    invSortedTarget = (sortedTarget == 0).astype('float32')
    Nneg = invSortedTarget.sum()
    fp = np.cumsum(invSortedTarget)
    tn = fp * -1 + Nneg
    accuracies = (tp + tn) / sortedTarget.shape[0]
    i = accuracies.argmax()
    print(i, tp[i], tn[i], sortedTarget.shape[0], accuracies[i])
    # calculate recall precision and F1
    Npos = sortedTarget.sum()
    fn = tp * -1 + Npos
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = (2*precision[i]*recall[i])/(precision[i]+recall[i])
    print("Best threshold", sortedErrors[i], "Accuracy:", accuracies[i], "Precision, Recall and F1 are %.5f %.5f %.5f" % (precision[i], recall[i], f1))
    print("TP, FP, TN, FN are %.5f %.5f %.5f %.5f" % (tp[i], fp[i], tn[i], fn[i]))
    return accuracies[i].astype(np.float32)
