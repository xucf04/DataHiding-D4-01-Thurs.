"""Binary steganalysis metrics for interleaved cover/stego scores."""

import numpy as np


def roc_summary(scores):
    """Return minimum balanced error, its threshold, and ROC AUC."""
    labels = np.arange(len(scores)) % 2
    n_cover = int((labels == 0).sum())
    n_stego = int((labels == 1).sum())
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    true_positive = np.r_[0, np.cumsum(sorted_labels == 1)[ends]]
    false_positive = np.r_[0, np.cumsum(sorted_labels == 0)[ends]]
    thresholds = np.r_[np.nextafter(float(sorted_scores[0]), np.inf), sorted_scores[ends]]
    pe = 0.5 * (false_positive / n_cover + (n_stego - true_positive) / n_stego)
    best = int(np.argmin(pe))
    auc = float(np.trapz(true_positive / n_stego, false_positive / n_cover))
    return {"pe": float(pe[best]), "threshold": float(thresholds[best]), "auc": auc}


def confusion_at(scores, threshold):
    labels = np.arange(len(scores)) % 2
    predicted = scores >= threshold
    tn = int(((labels == 0) & ~predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    tp = int(((labels == 1) & predicted).sum())
    return {"matrix": [[tn, fp], [fn, tp]],
            "pe": 0.5 * (fp / (tn + fp) + fn / (fn + tp))}
