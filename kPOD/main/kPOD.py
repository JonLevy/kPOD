# imports for mathematical functions
import numpy as np
from numpy import nanmean, nan, isnan
import sys
from scipy.spatial import distance
import pandas as pd

# import helper methods
from ..utils.initial_helpers import __initialize
from ..utils.utils import (
    __check_convergence, 
    __cluster_assignment, 
    __fill_data, 
    __move_centroids
)

def k_pod(data, n_clusters,max_iter=300,tol=0):
    """ Compute cluster centers and predict cluster index for sample containing missing data.

    Parameters
    ----------
    data: {array-like, sparse matrix} of shape (N, P)
        Data to predict clusters for.

    n_clusters: int
        The number of clusters to choose
    
    max_iter: int
        Maximum number of iterations to be run, default 300.

    tol: float
        Tolerance level for movement in cluster centroids, default 0.

    Returns 
    -------
    labels: ndarray of shape (N,)
        Index of the cluster each sample belongs to.

    """
    # convert data to numpy array
    data = np.array(data)
    
    # assign initial variables
    N = data.shape[0]
    P = data.shape[1]
    K = n_clusters
    num_iters = 0   

    # collect missing indiices
    MISSING_DATA = data.copy()

    # initialize past centroids
    past_centroids = []
    cluster_centers = []
    cluster_assignment = []
    indexes_with_nan = set()
    for index, line in enumerate(data):
        if any([isnan(i) for i in line]):
            indexes_with_nan.add(index)

    # loop through max iterations of kPOD
    while num_iters < max_iter:

        """
        STEP 1: Imputation of missing values
        fill with the mean of the cluster (centroid)
        """

        # if it has been multiple iterations, fill with algorithm
        if num_iters > 0:

            cluster_assignment, cluster_centers = recluster_if_all_records_have_nan(
                MISSING_DATA, cluster_centers, cluster_assignment, indexes_with_nan)

            # fill data after first iteration
            filled_data = __fill_data(MISSING_DATA, cluster_centers, cluster_assignment)

            # save data as np array
            filled_data = np.array(filled_data)

            if len(cluster_centers) != K:  # cluster have been removed, so re-initialize
                cluster_centers = __initialize(filled_data, K)

        # fill with initial imputation if first iteration 
        else:

            # initial imputation
            data_frame = pd.DataFrame(data)
            filled_data = np.array(data_frame.fillna(nanmean(data)))

            # initialize cluster centers so other methods work properly
            cluster_centers = __initialize(filled_data, K)
        
        """
        STEP 2: K-Means Iteration
        """

        # Cluster Assignment
        cluster_assignment = __cluster_assignment(filled_data, cluster_centers, N, K)

        # Move centroids
        cluster_centers = __move_centroids(filled_data, cluster_centers, cluster_assignment, N, K)
        
        """
        STEP 3: Check for convergence
        """

        centroids_complete = __check_convergence(cluster_centers, past_centroids, tol, num_iters)  

        # set past centroids to current centroids  
        past_centroids = cluster_centers.copy()

        # increase counter of iterations
        num_iters += 1

        # if k means is complete, end algo
        if centroids_complete == n_clusters:
            break

    # return assignments and centroids
    cluster_ret = {"ClusterAssignment" : cluster_assignment, "ClusterCenters" : cluster_centers}
    
    cluster_return  = (cluster_assignment, cluster_centers)
    return cluster_return

def recluster_if_all_records_have_nan(
        data, cluster_centers, cluster_assignment, indexes_with_nan):
    """
    Clusters where all the records have a nan are considered invalid.
    Such clusters are removed, and the records assigned to them are assigned to
    their nearest cluster center, measured with only the non-nan attributes.
    """

    def get_distance(point, center):
        'Simple euclidian distance'
        dist = 0
        for index, val in enumerate(point):
            dist += (val - center[index]) ** 2
        return dist ** .5

    def find_closest_center_with_nan(point, centers, exclude_clusters):
        """
        Find the center closest the point (based on non-nan attributes),
        and other than excluded_clusters.
        """
        new_centers = [list(i) for i in centers]

        remove_attribute_indexes = []
        for index, val in enumerate(point):
            if isnan(val):
                remove_attribute_indexes.append(index)
        remove_attribute_indexes.sort(reverse=True)

        for val in new_centers:
            for att in remove_attribute_indexes:
                del val[att]

        point = [i for i in point if not isnan(i)]

        closest = None
        min_dist = float('inf')
        for index, nc in enumerate(new_centers):
            if index in exclude_clusters:
                continue
            if get_distance(point, nc) < min_dist:
                closest = index
        return closest

    # figure out if there are clusters to remove `exclude_clusters`
    cluster_assignment = list(cluster_assignment)
    cluster_count_excluding_nans = {}
    for index, val in enumerate(cluster_assignment):
        if index in indexes_with_nan:
            continue
        cluster_count_excluding_nans[val] = cluster_count_excluding_nans.get(
            val, 0) + 1

    all_clusters = set(cluster_assignment)

    exclude_clusters = all_clusters - set(cluster_count_excluding_nans.keys())

    # reassign excluded_clusters to other clusters
    for index in indexes_with_nan:
        closest_cluster = find_closest_center_with_nan(
            data[index], cluster_centers, exclude_clusters)
        cluster_assignment[index] = np.float64(closest_cluster)

    # remove the cluster_centers
    exclude_clusters = list(exclude_clusters)
    exclude_clusters.sort(reverse=True)
    cluster_centers = list(cluster_centers)
    for ex in exclude_clusters:
        del cluster_centers[int(ex)]

        for index, val in enumerate(cluster_assignment):
            # decrement cluster values, so there are no gaps
            if val > ex:
                cluster_assignment[index] = val - 1

    return np.asarray(cluster_assignment), np.asarray(cluster_centers)
