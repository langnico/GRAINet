import numpy as np
from keras import losses
from keras import backend as K

# set up printing values from numpy arrays to not use exponential representation
np.set_printoptions(suppress=True)


def mean_IOU(y_true, y_pred):
    """
    Intersection over Union. Returns negative IoU to minimize the loss.
    """
    min_elements = K.tf.where(K.tf.less_equal(y_true, y_pred), y_true, y_pred)  # get the element-wise minimum
    max_elements = K.tf.where(K.tf.greater_equal(y_true, y_pred), y_true, y_pred)  # get the element-wise maximum

    intersection = K.sum(min_elements, axis=-1)
    union = K.sum(max_elements, axis=-1)

    iou = intersection / union  # this should be a tensor of shape: (batchsize, 1)

    return K.mean(-iou)


def KL(y_true, y_pred):
    """
    Kullback-Leibler divergence.
    Epsilon is used here to avoid conditional code for checking that neither P nor Q is equal to 0.
    """
    epsilon = 1e-07
    y_true = np.clip(y_true, epsilon, 1.)
    y_pred = np.clip(y_pred, epsilon, 1.)
    divergence = np.sum(y_true * np.log(y_true / y_pred))
    return divergence


def reverseKL(y_true, y_pred):
    """
    Reverse Kullback-Leibler divergende
    """
    return losses.kullback_leibler_divergence(y_pred, y_true)


def JSD(p, q):
    """
    Jensen-Shannon divergence: A smoothed and symmetric version of the KL divergence.
    """
    m = 0.5 * (p + q)
    return 0.5 * losses.kullback_leibler_divergence(p, m) + 0.5 * losses.kullback_leibler_divergence(q, m)


def emd(y_true, y_pred):
    """
    Earth mover's distance (EMD) for 1D-histograms (also known as Wasserstein metric).
    Args:
        y_true: ground truth histograms (batch_size, bins)
        y_pred: predicted histograms (batch_size, bins)

    Returns: mean EMD over batch

    """
    cdf_true = K.cumsum(y_true, axis=1)
    cdf_pred = K.cumsum(y_pred, axis=1)

    return K.mean(K.sum(K.abs(cdf_true - cdf_pred), axis=1))


def get_mean_size_squared():
    """
    helper to calculate volume weights
    :return: volume proxy to weight bin errors.
    """
    # limits of all grain classes
    edges = np.array(
        [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80,
         1.0, 1.2, 1.5, 2.0], dtype=float)

    n_bins = len(edges) - 1
    # mean diameter of the class i
    dmi = np.zeros(n_bins)
    for d in range(0, n_bins):
        dmi[d] = (edges[d+1] + edges[d]) / 2.

    # percentage mass of class i, before additional empirical corrections
    mean_bin_volume = np.power(dmi, 2)

    return mean_bin_volume


def mae_weighted(y_true, y_pred):
    weights = get_mean_size_squared()
    weights = weights / float(np.sum(weights))
    weights = K.constant(weights)
    diff = y_pred - y_true
    loss = K.mean(K.abs(weights * diff))
    return loss


def mse_weighted(y_true, y_pred):
    weights = get_mean_size_squared()
    weights = weights / float(np.sum(weights))
    weights = K.constant(weights)
    diff = y_pred - y_true
    loss = K.mean(K.square(weights * diff))
    return loss


# helper functions to compute metrics from numpy arrays.

def calculate_iou(y_true, y_pred):
    """
    Intersection over Union for numpy arrays.
    """
    number_of_bins = y_true.shape[0]
    intersection = 0
    union = 0
    for i in range(number_of_bins):
        intersection = intersection + min(y_true[i], y_pred[i])
        union = union + max(y_true[i], y_pred[i])
    iou = intersection / union
    return iou


def calculate_emd(y_true, y_pred):
    """
    Earth mover's distance (EMD) for 1D pdfs numpy arrays.
    """
    cdf_true = np.cumsum(y_true)
    cdf_pred = np.cumsum(y_pred)

    return np.sum(np.abs(cdf_true - cdf_pred))