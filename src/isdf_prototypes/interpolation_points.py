""" Module for functions that find interpolation grid points:

* QR decomposition with column pivoting
* Weighted k-means clustering
"""

from typing import List, Tuple

import numpy as np
from ordered_set import OrderedSet
from scipy.spatial.distance import cdist

cluster_type = List[List[int]]


def is_subgrid_on_grid(subgrid: np.ndarray, grid: np.ndarray, tol: float) -> list:
    """Return indices of subgrid points not present in grid.

    Parallelisation:
    * Won't benefit from Numba, as only called once
    * Loop shouldn't be large enough to parallelise with MPI
      - One could thread it
    * Function might benefit from cupy

    :param subgrid:
    :param grid:
    :return:
    """
    N_sub = subgrid.shape[0]
    indices = []
    for i in range(N_sub):
        centred_on_point_i = np.linalg.norm(grid - subgrid[i], axis=1)
        matched_indices = np.argwhere(centred_on_point_i <= tol)
        if matched_indices.size == 0:
            indices.append(i)
    return indices


def assign_points_to_centroids(grid_points: np.ndarray, centroids: np.ndarray) -> cluster_type:
    """Assign each grid point to the closest centroid. A centroid and its set of nearest
    grid points defines a cluster.

    Refactor:
    * Could work on the distance matrix for all points
    * Could simplify such that I just use argmin. Not likely that points will be equidistant
      in all but symmetric, toy problems

    Parallelisation Strategies:
    * Could benefit from MPI distribution of loop
    * ~Could benefit from NUMBA, as called multiple times~   cdist is scipy, not numpy
      - Would have to @overload(scipy.spatial.distance.cdist) and supply my own compiled implementation
    * Could benefit from cupy as *some* numpy operations are used

    :param grid_points: Grid point
    :param centroids: Centroid points

    :return: clusters: List of length N interpolation points. Each element corresponds to a cluster,
    which indexes all grid points associated with it
    """
    N_r, dim = grid_points.shape
    N_interp = centroids.shape[0]
    clusters: cluster_type = [[] for _ in range(N_interp)]

    # Computing distance matrix per grid point because I want to distribute this loop
    # (keeping a fortran implementation in mind)
    for ir in range(0, N_r):
        # Need to retain a 2D array for cdist
        point = grid_points[ir, :][None, :]
        # distance matrix has shape (1, N_interp)
        distance_matrix = cdist(point, centroids).reshape(-1)
        # min(|r_{ir} - initial_centroids|)

        # TODO(Alex) Reintroduce once testing of k-means is done
        # min_index = np.argmin(distance_matrix)
        # # If two or more elements are equally minimal, argmin will always return the first instance
        # # Instead, we find all equally minimum indices, then choose one at random
        # min_indices = np.argwhere(distance_matrix == distance_matrix[min_index])[:, 0]
        # icen = np.random.choice(min_indices)

        icen = np.argmin(distance_matrix)
        clusters[icen].append(ir)
    return clusters


def update_centroids(grid_points, f_weight, clusters: cluster_type) -> np.ndarray:
    """Compute a new set of initial_centroids

    We have as many clusters as we do initial_centroids

    :param grid_points: Grid
    :param f_weight: Weight function
    :param clusters: Grid point indices associated with each cluster
    :return: updated_centroids: Updated initial_centroids
    """
    N_interp = len(clusters)
    dim = grid_points.shape[1]
    updated_centroids = np.empty(shape=(N_interp, dim))

    for icen in range(0, N_interp):
        # Indices of grid points associated with this cluster
        grid_indices = np.asarray(clusters[icen])
        # Part of the weight function associated with this cluster
        weights = f_weight[grid_indices]
        weighted_pos = np.sum(grid_points[grid_indices] * weights[:, np.newaxis], axis=0)
        updated_centroids[icen, :] = weighted_pos / np.sum(weights)
    return updated_centroids


def points_are_converged(updated_points, points, tol, verbose=False) -> bool:
    """Given the difference in two sets of points, determine whether the updated
    points are sufficiently close to the prior points.

    :param updated_points:
    :param points:
    :return:
    """
    vector_diffs = updated_points - points
    norm = np.linalg.norm(vector_diffs, axis=1)
    indices = np.where(norm > tol)[0]
    converged = indices.size == 0

    if verbose:
        N = updated_points.shape[0]
        if converged:
            print("Convergence: All points converged")
        else:
            print(f"Convergence: {len(indices)} points out of {N} are not converged:")
            print("# Current Point    Prior Point    |ri - r_{i-1}|   tol")
            for i in indices:
                print(updated_points[i, :], points[i, :], norm[i], tol)

    return converged


def verbose_print(str, verbose=True):
    if verbose:
        print(str)


def kmeans_seeding(grid: np.ndarray, n_centroids: int) -> np.ndarray:
    """ Seeding for k-means.

    An implementation of the seeding in k-means++, as given in Algorithm 1 of 10.4230/LIPIcs.ESA.2020.18
    and based on the original implementation in https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf

    This implementation would be transcribable to fortran, except for the use of Ordered Sets
    for index maps. There, one could instead reallocate index arrays like:

    ```fortran
    Program assign
        implicit none
        integer, allocatable :: indices(:)
        integer :: b(8)
        integer :: icen

        b = [1, 2, 3, 4, 5, 6, 7, 8]
        ! Drop index 5
        icen = 5
        indices = [b(:icen-1), b(icen+1:)]
        write(*, *) indices

    End Program assign
    ```
    It's just not very efficient, but it is readable. I don't see a way around it unless one uses if statements.

    :return:
    """
    n_points, ndim = grid.shape
    centroids = np.empty(shape=(n_centroids, ndim))
    pdf = np.zeros(shape=n_points)

    # Sample first centroid independently and uniformly at random, from X
    # Size=1 returns an iterable, required as the arg for OrderedSet
    icen = np.random.choice(n_points, size=1, replace=False)
    centroid_indices = OrderedSet(icen)
    grid_indices = OrderedSet([i for i in range(0, n_points)]) - centroid_indices
    centroids[0] = grid[icen, :]

    for i in range(1, n_centroids):
        for ir in grid_indices:
            # p(x) = min_{c \in C} ||c - x||^2
            pdf[ir] = np.min(np.sum((centroids[0:i, :] - grid[ir, :]) ** 2, axis=1))

        gindx = list(grid_indices)
        # p(x) = p(x) / \sum_{x' \in X} min_{c \in C}  ||c - x'||^2
        pdf[gindx] = pdf[gindx] / sum(pdf[gindx])
        # Sample a point c_i from X, where every x\inX has probability p(x)
        icen = np.random.choice(gindx, replace=False, p=pdf[gindx])
        # Update indices
        centroid_indices.append(icen)
        grid_indices -= OrderedSet([icen])
        # Update centroids
        centroids[i, :] = grid[icen, :]
    return centroids


def weighted_kmeans(
        grid_points: np.ndarray,
        f_weight: np.ndarray,
        initial_centroids: np.ndarray,
        n_iter=200,
        centroid_tol=1.0e-6,
        safe_mode=False,
        verbose=True,
) -> Tuple[np.ndarray, int]:
    """ Perform weighted k-means clustering.

    TODOs:
     * Describe algorithm
     * Add latex
     * Try version of routine with MPI
     * Try version of routine with JIT
     * Try version of routine with cupy and numba

    :param grid_points: Real space grid
    :param f_weight: Weight function
    :param initial_centroids: Initial centroids. A good choice is a set of N randomly (or uniformly) distributed
    points.
    The size of this array defines the number of interpolating grid points, N.
    These points must be part of the set of grid_points (?)
    :param n_iter: Number of iterations to find optimal centroids
    :return: interpolation_points: Grid points for interpolating vectors, as defined by optimised centroids
    """
    if n_iter < 1:
        raise ValueError("n_iter must be non-negative.")

    if grid_points.ndim != 2:
        raise ValueError("Expect grid to be shaped (N_points, dim).")

    if safe_mode:
        indices = is_subgrid_on_grid(initial_centroids, grid_points, 1.0e-6)
        n_off_grid = len(indices)
        if n_off_grid > 0:
            print(
                f"{n_off_grid} out of {initial_centroids.shape[0]} initial_centroids are not defined on the real-space grid"
            )
            print("# Index     Point")
            for i in indices:
                print(i, initial_centroids[i, :])
            raise ValueError()

    N_r, dim = grid_points.shape

    if f_weight.shape[0] != N_r:
        err_msg = (
            f"Number of sampling points defining the weight function, {f_weight.shape[0]}, differs to the size of the grid {N_r}\n. "
            "Weight function must be defined on the same real-space grid as grid_points"
        )
        raise ValueError(err_msg)

    verbose_print("Centroid Optimisation", verbose)
    centroids = np.copy(initial_centroids)

    for t in range(0, n_iter):
        clusters = assign_points_to_centroids(grid_points, centroids)
        updated_centroids = update_centroids(grid_points, f_weight, clusters)
        verbose_print(f"Step {t}", verbose)
        converged = points_are_converged(updated_centroids, centroids, centroid_tol, verbose=verbose)
        if converged:
            return updated_centroids, t
        centroids = updated_centroids

    return centroids, t
