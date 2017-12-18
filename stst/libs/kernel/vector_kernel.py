"""
Ref:
1. http://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation
2. Linear Kernel
3. Non-Linear Kernel

Code: https://github.com/scikit-learn/scikit-learn/blob/412996f/sklearn/metrics/pairwise.py#L836


# Helper functions - distance
PAIRWISE_DISTANCE_FUNCTIONS = {
    # If updating this dictionary, update the doc in both distance_metrics()
    # and also in pairwise_distances()!
    'cosine': cosine_distances,
    'euclidean': euclidean_distances,
    'manhattan': manhattan_distances,
}

# These distances recquire boolean arrays, when using scipy.spatial.distance
PAIRWISE_BOOLEAN_FUNCTIONS = [
    'dice',
    'jaccard',
    'kulsinski',
    'matching',
    'rogerstanimoto',
    'russellrao',
    'sokalmichener',
    'sokalsneath',
    'yule',
]


# Helper functions - distance
PAIRWISE_KERNEL_FUNCTIONS = {
    # If updating this dictionary, update the doc in both distance_metrics()
    # and also in pairwise_distances()!
    'additive_chi2': additive_chi2_kernel,
    'chi2': chi2_kernel,
    'linear': linear_kernel,
    'polynomial': polynomial_kernel,
    'poly': polynomial_kernel,
    'rbf': rbf_kernel,
    'laplacian': laplacian_kernel,
    'sigmoid': sigmoid_kernel,
    'cosine': cosine_similarity, }


"""

import numpy as np
import scipy.stats
from sklearn.metrics.pairwise import additive_chi2_kernel

''' Linear Kernel '''


def cosine_distance(v1, v2):
    """

    :param v1: numpy vector
    :param v2: numpy vector
    :return:
    """
    v1, v2 = check_pairwise_vector(v1, v2)

    # cosine = np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))
    #
    # if np.isnan(cosine):
    #    cosine = 1.

    cosine = (v1 * v2).sum()

    return 1. - cosine


def manhattan_distance(v1, v2):
    v1, v2 = check_pairwise_vector(v1, v2)

    diff = v1 - v2
    K = np.abs(diff).sum()

    return K


def euclidean_distance(v1, v2):
    v1, v2 = check_pairwise_vector(v1, v2)

    diff = v1 - v2
    K = np.sqrt((diff ** 2).sum())

    return K


def chebyshev_distance(v1, v2):
    v1, v2 = check_pairwise_vector(v1, v2)

    diff = v1 - v2
    K = np.abs(diff).max()

    return K


LinearKernel = {
    'cosine': cosine_distance,
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance,
    'chebyshev_distance': chebyshev_distance
}

''' Stat Kernel '''


def pearsonr_stat(v1, v2):
    v1, v2 = check_pairwise_vector(v1, v2)

    r, prob = scipy.stats.pearsonr(v1, v2)

    return r


def spearmanr_stat(v1, v2):
    v1, v2 = check_pairwise_vector(v1, v2)

    r, prob = scipy.stats.spearmanr(v1, v2)

    return r


def kendalltau_stat(v1, v2):
    v1, v2 = check_pairwise_vector(v1, v2)

    r, prob = scipy.stats.kendalltau(v1, v2)

    return r


StatKernel = {
    'pearsonr': pearsonr_stat,
    'spearmanr': spearmanr_stat,
    'kendalltau': kendalltau_stat
}

''' Non Linear Kernel '''


def additive_chi2(v1, v2):
    """
     k(x, y) = -Sum [(x - y)^2 / (x + y)]
    :param v1:
    :param v2:
    :return:
    """
    v1, v2 = check_pairwise_vector(v1, v2)

    X, Y = v1.reshape(v1.shape[0], -1), v2.reshape(v2.shape[0], -1)
    K = additive_chi2_kernel(X, Y)[0][0]
    return K


def chi2(v1, v2, gamma=1.):
    """
    k(x, y) = exp(-gamma Sum [(x - y)^2 / (x + y)])
    :param v1:
    :param v2:
    :return:
    """
    v1, v2 = check_pairwise_vector(v1, v2)

    K = additive_chi2(v1, v2)
    K *= gamma
    K = np.exp(K)
    return K


def polynomial(v1, v2, degree=3, gamma=None, coef0=1):
    """
    K(X, Y) = (gamma <X, Y> + coef0)^degree
    :param v1: numpy vector
    :param v2:
    :return:
    """
    v1, v2 = check_pairwise_vector(v1, v2)

    if gamma is None:
        gamma = 1.0 / v1.shape[0]

    K = np.dot(v1, v2)
    K *= gamma
    K += coef0
    K **= degree
    return K


def rbf(v1, v2, gamma=None):
    """
     K(x, y) = exp(-gamma ||x-y||^2)
    :param v1:
    :param v2:
    :param gamma:
    :return:
    """
    v1, v2 = check_pairwise_vector(v1, v2)

    if gamma is None:
        gamma = 1.0 / v1.shape[0]

    K = euclidean_distance(v1, v2)
    K *= -gamma
    K = np.exp(K)
    return K


def laplacian(v1, v2, gamma=None):
    """
     K(x, y) = exp(-gamma ||x-y||_1)
    :param v1:
    :param v2:
    :return:
    """
    v1, v2 = check_pairwise_vector(v1, v2)

    if gamma is None:
        gamma = 1.0 / v1.shape[0]

    K = manhattan_distance(v1, v2)
    K *= -gamma
    K = np.exp(K)
    return K


def sigmoid(v1, v2, gamma=None, coef0=1):
    """
    K(X, Y) = tanh(gamma <X, Y> + coef0)
    :param v1:
    :param v2:
    :return:
    """
    v1, v2 = check_pairwise_vector(v1, v2)

    if gamma is None:
        gamma = 1.0 / v1.shape[0]

    K = np.dot(v1, v2)
    K *= gamma
    K += coef0
    K = np.tanh(K)  # compute tanh in-place
    return K


NonLinearKernel = {
    # 'additive_chi2': additive_chi2,
    # 'chi2': chi2,
    'polynomial': polynomial,
    'rbf': rbf,
    'laplacian': laplacian,
    'sigmoid': sigmoid
}


def check_pairwise_vector(v1, v2):
    if isinstance(v1, list):
        v1 = np.array(v1)
    if isinstance(v2, list):
        v2 = np.array(v2)

    if v1.shape != v2.shape:
        raise ValueError("v1 and v2 should be of same shape. They were "
                         "respectively %r and %r long." % (v1.shape, v2.shape))
    return v1, v2


def get_linear_kernel(v1, v2):
    linear_kernel_feats = []
    linear_kernel_names = []
    for function_name in LinearKernel:
        function = LinearKernel[function_name]
        K = function(v1, v2)
        linear_kernel_feats.append(K)
        linear_kernel_names.append(function_name)
    return linear_kernel_feats, linear_kernel_names


def get_stat_kernel(v1, v2):
    feats = []
    names = []
    for function_name in StatKernel:
        function = StatKernel[function_name]
        K = function(v1, v2)
        feats.append(K)
        names.append(function_name)

    "nan"
    feats = [0.0 if np.isnan(feat) else feat for feat in feats]
    return feats, names


def get_non_linear_kernel(v1, v2):
    non_linear_feats = []
    non_linear_names = []
    for function_name in NonLinearKernel:
        function = NonLinearKernel[function_name]
        K = function(v1, v2)
        non_linear_feats.append(K)
        non_linear_names.append(function_name)
    return non_linear_feats, non_linear_names


def get_all_kernel(v1, v2):
    """
    
    :param v1: 
    :param v2: 
    :return: 
    
    example:
        X = [0, 1]
        Y = [1, 0]
     
        ('euclidean', 2)
        ('cosine', 1.0)
        ('manhattan', 2)
        ('spearmanr', -0.99999999999999989)
        ('kendalltau', -0.99999999999999989)
        ('pearsonr', -1.0)
        ('additive_chi2', -1.0)
        ('sigmoid', 0.76159415595576485)
        ('chi2', 0.36787944117144233)
        ('laplacian', 0.36787944117144233)
        ('polynomial', 1.0)
        ('rbf', 0.36787944117144233)
    """

    linear_kernel_feats, linear_kernel_names = get_linear_kernel(v1, v2)
    stat_kernel_feats, stat_kernel_names = get_stat_kernel(v1, v2)
    non_linear_feats, non_linear_names = get_non_linear_kernel(v1, v2)

    all_feats = linear_kernel_feats + stat_kernel_feats + non_linear_feats
    all_names = linear_kernel_names + stat_kernel_names + non_linear_names

    return all_feats, all_names


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


if __name__ == '__main__':

    v1 = np.array([0, 3], dtype=np.float32)
    v2 = np.array([4, 0], dtype=np.float32)
    print(euclidean_distance(v1, v2))
    print(chebyshev_distance(v1, v2))
    print(cosine_distance([0, 0], [0, 0]))

    X = [0, 1]
    Y = [1, 0]

    X = np.array(X)
    Y = np.array(Y)

    feats, names = get_all_kernel(X, Y)
    for name, feat in zip(names, feats):
        print(name, feat)
        '''
        ('euclidean', 2)
        ('cosine', 1.0)
        ('manhattan', 2)
        ('spearmanr', -0.99999999999999989)
        ('kendalltau', -0.99999999999999989)
        ('pearsonr', -1.0)
        ('additive_chi2', -1.0)
        ('sigmoid', 0.76159415595576485)
        ('chi2', 0.36787944117144233)
        ('laplacian', 0.36787944117144233)
        ('polynomial', 1.0)
        ('rbf', 0.36787944117144233)
        '''
