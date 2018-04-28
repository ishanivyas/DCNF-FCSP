import numpy as np
from util import clamp01

def project_back_label(labels, norm=None):
    """Back-project labels."""
    if norm is None:
        return labels

    scaling      = norm.max_d - norm.min_d
    offset       = norm.min_d
    labels_trans = clamp01(labels)*scaling + offset

    if norm.norm_type == 2:
        k = 2
    elif norm.norm_type == 3:
        k = 10
    else:
        return labels_trans

    return np.power(np.full_like(labels, k), labels_trans)
