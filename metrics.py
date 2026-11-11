import numpy as np

def exact_metrics(scores, labels, zero_negative, zero_positive):
    order = np.argsort(scores, kind='stable')[::-1]
    sorted_score = np.asarray(scores[order])
    sorted_label = np.asarray(labels[order], dtype=np.uint8)
    del order
    tp = np.cumsum(sorted_label, dtype=np.int64)
    ends = np.r_[np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), len(sorted_score) - 1]
    thresholds = sorted_score[ends].copy()
    true = tp[ends].astype(np.float64)
    false = (ends + 1).astype(np.float64) - true
    del tp, ends, sorted_score, sorted_label
    if thresholds[-1] == 0:
        true[-1] += zero_positive
        false[-1] += zero_negative
    else:
        thresholds = np.r_[thresholds, np.float32(0)]
        true = np.r_[true, true[-1] + zero_positive]
        false = np.r_[false, false[-1] + zero_negative]
    positives = true[-1]
    negatives = false[-1]
    assert positives > 0 and negatives > 0
    recall = true / positives
    precision = true / (true + false)
    ap = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    auc = float(np.trapz(np.r_[0, recall], np.r_[0, false / negatives]))
    f1 = 2 * true / (positives + true + false)
    index = len(f1) - 1 - int(np.argmax(f1[::-1]))
    return {'AP': ap, 'ROCAUC': auc, 'Threshold': float(thresholds[index]), 'global_optimal_dice': float(f1[index]), 'positive_voxels': int(positives), 'negative_voxels': int(negatives)}
