import numpy as np

def compute_cost_matrix_cosine_distance(X, Y, alpha=1):
    """Compute the cost matrix of two feature sequences
    using the formula from Matlab SynchToolbox. (Script computeC.m)

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        X: Sequence 1
        Y: Sequence 2
        metric: Cost metric, a valid strings for scipy.spatial.distance.cdist

    Returns:
        C: Cost matrix
    """
    cosMeasMin=1
    cosMeasMax = 2

    X, Y = np.atleast_2d(X, Y)
    #C = scipy.spatial.distance.cdist(X.T, Y.T, metric='cosine') + alpha
    C = (1 - np.dot(X.T,Y)) + alpha
    return C
