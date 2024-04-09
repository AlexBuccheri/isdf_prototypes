from pathlib import Path

import numpy as np
import pytest

from isdf_prototypes.math_ops import face_splitting_product
from . import tests_root
from .molecular_calculation import (MinimalBasisBenzene, generate_ks_states, compute_real_space_density,
                                    compute_real_space_grid)


from isdf_prototypes.interpolation_points_via_qr import randomly_sample_product_matrix, interpolation_points_via_qrpivot


@pytest.fixture(scope="module")
def benzene_dft():
    """ Set up a completed KS DFT instance for benzene
    """
    calculation = MinimalBasisBenzene()
    calculation.molecule()

    # Annoyingly cannot pickle calculation, so just save all required quantities
    file = tests_root / Path("benzene_wfs.npy")
    if file.exists():
        print("Loading wfs")
        wfs = np.load(file)
    else:
        # In first instance run the DFT calc
        calculation.run_dft()
        wfs = generate_ks_states(calculation)
        np.save(file, wfs)

    file = tests_root / Path("benzene_rho.npy")
    if file.exists():
        print("Loading rho")
        rho = np.load(file)
    else:
        cube_file = tests_root / Path("cube_files/density.cube")
        rho = compute_real_space_density(calculation, cube_file.as_posix())
        np.save(tests_root / Path("benzene_rho"), rho)

    file = tests_root / Path("benzene_grid.npy")
    if file.exists():
        print("Loading grid")
        grid = np.load(file)
    else:
        grid = compute_real_space_grid(calculation)
        np.save(file, grid)

    return calculation.mol, wfs, rho, grid


def test_zct_expansion(benzene_dft):
    # Construct ZC^T and compare to P dot P.

    mol, wfs, rho, grid = benzene_dft

    n_grid_points = 1000
    n_occ = 22

    # If this isn't true, the parameters have changed and the mock data needs regenerating
    assert wfs.shape == (n_grid_points, n_occ), "Expect (n_grid_points, n_occ_states)"
    assert rho.shape == (n_grid_points,), "Expect (n_grid_points)"
    assert grid.shape == (n_grid_points, 3), "Expect (n_grid_points, 3)"

    # Get interpolation points
    n_interp = 110
    z_sampled = randomly_sample_product_matrix(wfs, n_interp=n_interp, random_seed=42)
    assert grid.shape[0] == z_sampled.shape[0]

    n_interp = z_sampled.shape[1]
    assert n_interp == 100, "Number of interpolation vectors rounded to closest value with integer root"
    indices = interpolation_points_via_qrpivot(z_sampled)
    assert len(indices) == n_interp

    # Z
    z = face_splitting_product(wfs)
    n_product_basis = n_occ ** 2
    assert n_product_basis == 484
    assert z.shape == (n_grid_points, n_product_basis)

    # C
    c = z[indices, :]
    assert c.shape == (n_interp, n_product_basis)

    # Contraction of Z and C^T matrices
    ref_zct = z @ c.T
    assert ref_zct.shape == (n_grid_points, n_interp)

    # Compare to separable formulation, which is implemented in `construct_interpolation_vectors_matrix`
    zct = (wfs @ wfs[indices, :].T) * (wfs @ wfs[indices, :].T)
    assert zct.shape == (n_grid_points, n_interp)

    assert np.allclose(ref_zct, zct), "separable formulation consistent with matrix-matrix product"



# SKLearn routines continuous centroids, rather than centroids bound to the discrete grid
# As such, do testing with a different algorithm for grid points.
# def interpolation_points(grid, weights, n_clusters, random_state=10) -> tuple:
#     """ Interpolation points
#
#     :param grid: Grid (n_points, 3)
#     :param weights: Weights (n_points)
#     :param n_clusters: Number of clusters/interpolation points
#     :param random_state: Using an int will produce the same results across different calls.
#            Popular integer random seeds are 0 and 42.
#     :return: clusters, indices
#     """
#     assert grid.shape[0] == weights.shape[0], "Weights and grid must be defined for same number of points"
#     k_means = KMeans(n_clusters=n_clusters,
#                      init='k-means++',
#                      n_init='auto',
#                      max_iter=300,
#                      tol=0.0001,
#                      random_state=random_state,
#                      copy_x=True,
#                      algorithm='lloyd')
#     k_means.fit(grid, sample_weight=weights)
#
#     clusters = k_means.cluster_centers_
#     assert len(clusters) == n_clusters, "Ensure kmeans returns expected number of clusters"
#
#     # For each cluster, find the closest grid point to it
#     # grid_cluster_points = np.empty_like(clusters)
#     # indices = np.empty(shape=n_clusters)
#     # for j, point in enumerate(clusters):
#     #     abs_diff = np.abs(grid - point)
#     #     i = np.argmin(abs_diff.sum(axis=1))
#     #     grid_cluster_points[j, :] = grid[i, :]
#     #     indices[j] = i
#
#     # Can't use this.
#     # Grid index of each cluster
#     indices = find_subgrid_in_grid_single_loop(grid, clusters)
#     print(indices)
#     assert len(indices) == n_clusters, "Number of indices of subgrid points doesn't match the number of clusters"
#
#     return clusters, indices
