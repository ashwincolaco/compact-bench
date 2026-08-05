"""Small pure-python metrics: AUROC and ECE (no sklearn dependency)."""


def auroc(scores, labels):
    """Area under ROC via the rank statistic (Mann-Whitney U).

    scores: predicted score per item (higher = predicted positive).
    labels: 1/0 ground truth. Returns None if one class is absent.
    """
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def ece(confidences, labels, n_bins=10):
    """Expected calibration error with equal-width bins over [0,1]."""
    if not confidences:
        return None
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, labels):
        i = min(int(c * n_bins), n_bins - 1)
        bins[i].append((c, y))
    total = len(confidences)
    err = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        err += (len(b) / total) * abs(conf - acc)
    return err
