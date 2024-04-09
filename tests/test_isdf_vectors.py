from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import KMeans

from . import tests_root
from .molecular_calculation import (MinimalBasisBenzene, generate_ks_states, compute_real_space_grid,
                                    compute_real_space_density, find_subgrid_in_grid, find_subgrid_in_grid_single_loop,
                                    compute_real_space_grid)


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


def interpolation_points(grid, weights, n_clusters, random_state=10) -> tuple:
    """ Interpolation points

    :param grid: Grid (n_points, 3)
    :param weights: Weights (n_points)
    :param n_clusters: Number of clusters/interpolation points
    :param random_state: Using an int will produce the same results across different calls.
           Popular integer random seeds are 0 and 42.
    :return: clusters, indices
    """
    assert grid.shape[0] == weights.shape[0], "Weights and grid must be defined for same number of points"
    k_means = KMeans(n_clusters=n_clusters,
                     init='k-means++',
                     n_init='auto',
                     max_iter=300,
                     tol=0.0001,
                     random_state=random_state,
                     copy_x=True,
                     algorithm='lloyd')
    k_means.fit(grid, sample_weight=weights)

    clusters = k_means.cluster_centers_
    assert len(clusters) == n_clusters, "Ensure kmeans returns expected number of clusters"

    # For each cluster, find the closest grid point to it
    # grid_cluster_points = np.empty_like(clusters)
    # indices = np.empty(shape=n_clusters)
    # for j, point in enumerate(clusters):
    #     abs_diff = np.abs(grid - point)
    #     i = np.argmin(abs_diff.sum(axis=1))
    #     grid_cluster_points[j, :] = grid[i, :]
    #     indices[j] = i

    # Can't use this.
    # Grid index of each cluster
    indices = find_subgrid_in_grid_single_loop(grid, clusters)
    print(indices)
    assert len(indices) == n_clusters, "Number of indices of subgrid points doesn't match the number of clusters"

    return clusters, indices


def test_find_subgrid_in_grid(benzene_dft):
    _, _, _, grid = benzene_dft


    # np.set_printoptions(precision=15)
    # print(grid[9, :])
    # print(grid[101, :])
    # print(grid[8, :])
    # print(grid[87, :])
    # print(grid[999, :])

    subgrid = np.array([[-7.626786546123851, -7.07273774366262 , 10.679261155133798],
                        [-5.915393190274166, -7.07273774366262 ,  5.143903500910294],
                        [-7.626786546123851, -7.07273774366262 ,  9.98734144835586 ],
                        [-7.626786546123851,  5.500833471849858,  9.295421741577922],
                        [ 7.775753656523316,  7.072529873788919, 10.679261155133798]])

    indices = find_subgrid_in_grid_single_loop(grid, subgrid)
    assert all(indices == [9, 101, 8, 87, 999])
    assert np.allclose(grid[indices], subgrid)


def test_zct_expansion(benzene_dft):
    # Construct ZC^T and compare to P dot P.

    mol, wfs, rho, grid = benzene_dft

    # assert wfs.shape == (1000, 22), "Expect (n_grid_points, n_occ_states)"
    # assert rho.shape == (1000,), "Expect (n_grid_points)"
    # assert grid.shape == (1000, 3), "Expect (n_grid_points, 3)"

    n_clusters = 100
    interp_points, indices = interpolation_points(grid, rho, n_clusters, random_state=10)
    assert interp_points.shape == (n_clusters, 3)
    assert indices.shape == (n_clusters, )
    assert np.allclose(grid[indices], interp_points)

    # # Z
    # z = face_splitting_product(occupied_ks_states)
    # n_product_basis = 484
    # assert n_product_basis == occupied_ks_states.shape[0] ** 2
    # assert z.shape == (1000, n_product_basis)
    #
    # # C
    # c = np.empty(shape=1)
    #
    # Contraction of Z and C^T matrices
    # zct = z @ c.T

