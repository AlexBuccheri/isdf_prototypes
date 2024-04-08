import pickle
from pathlib import Path
from typing import List

import numpy as np
import pytest
import sympy
from sklearn.cluster import KMeans

from . import tests_root
from isdf_prototypes.isdf_vectors import iteration_implementation_face_splitting_product, \
    face_splitting_product_single_loop, face_splitting_product

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


# @pytest.fixture(scope="module")
# def occupied_ks_states(benzene_dft) -> np.ndarray:
#     wfs = generate_ks_states(benzene_dft)
#     return wfs
#


# @pytest.fixture(scope="module")
# def occupied_ks_states(benzene_dft) -> np.ndarray:
#     wfs = generate_ks_states(benzene_dft)
#     return wfs
#
#
# @pytest.fixture(scope="module")
# def density(benzene_dft) -> np.ndarray:
#     # Put unnecessary cube files in their own subdir
#     cube_root = tests_root / Path("cube_files")
#     cube_root.mkdir(exist_ok=True)
#     cube_file = cube_root / "density.cube"
#     rho = compute_real_space_density(benzene_dft, cube_file.as_posix())
#     return rho
#
#
# @pytest.fixture(scope="module")
# def real_space_grid(benzene_dft) -> np.ndarray:
#     return compute_real_space_grid(benzene_dft)


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

    # Grid index of each cluster
    indices = find_subgrid_in_grid_single_loop(grid, clusters)
    assert len(indices) == n_clusters, "Number of indices of subgrid points doesn't match the number of clusters"

    return clusters, indices


# Helper function
def sympy_to_numpy(str_list: List[str], shape) -> np.ndarray:
    """ Convert a list of strings into a symbolic numpy array, with correct shape.
    :param str_list:
    :param shape:
    :return:
    """
    assert len(str_list) == np.prod(shape)
    array_sympy = [sympy.sympify(expr) for expr in str_list]
    array = np.array(array_sympy, dtype=object).reshape(shape)
    return array


def test_face_splitting_product():
    # Input array
    a = np.array(sympy.symbols('a11 a12 a13 '
                               'a21 a22 a23 '
                               'a31 a32 a33 '
                               'a41 a42 a42 '), object).reshape((4, 3))

    # Face-splitting product of a x a
    ref_symbols = ['a11**2', 'a11*a12', 'a11*a13', 'a11*a12', 'a12**2', 'a12*a13', 'a11*a13', 'a12*a13', 'a13**2',
                   'a21**2', 'a21*a22', 'a21*a23', 'a21*a22', 'a22**2', 'a22*a23', 'a21*a23', 'a22*a23', 'a23**2',
                   'a31**2', 'a31*a32', 'a31*a33', 'a31*a32', 'a32**2', 'a32*a33', 'a31*a33', 'a32*a33', 'a33**2',
                   'a41**2', 'a41*a42', 'a41*a42', 'a41*a42', 'a42**2', 'a42**2 ', 'a41*a42', 'a42**2 ', 'a42**2']
    ref_fs_product = sympy_to_numpy(ref_symbols, (4, 9))

    # Loop-based
    z1 = iteration_implementation_face_splitting_product(a, a)
    assert z1.shape == (4, 9)
    assert np.array_equal(z1, ref_fs_product), "Expect same symbolic elements"

    # Single loop
    z2 = face_splitting_product_single_loop(a, a)
    assert z2.shape == (4, 9)
    assert np.array_equal(z2, ref_fs_product), "Expect same symbolic elements"

    # No loops -> pure numpy
    z3 = face_splitting_product(a, a)
    assert z3.shape == (4, 9)
    assert np.array_equal(z3, ref_fs_product), "Expect same symbolic elements"


def test_face_splitting_product_with_different_num_states():
    a = np.array(sympy.symbols('a11 a12 a13 '
                               'a21 a22 a23 '
                               'a31 a32 a33 '
                               'a41 a42 a42 '), object).reshape((4, 3))

    b = np.array(sympy.symbols('b11 b12 '
                               'b21 b22 '
                               'b31 b32 '
                               'b41 b42 '), object).reshape((4, 2))

    assert a.shape[0] == b.shape[0], "a and b must have the same number of rows"
    product_basis_size = 6
    assert a.shape[1] * b.shape[1] == product_basis_size

    # Face-splitting product of a x b
    ref_symbols = ['a11*b11', 'a12*b11', 'a13*b11', 'a11*b12', 'a12*b12', 'a13*b12',
                   'a21*b21', 'a22*b21', 'a23*b21', 'a21*b22', 'a22*b22', 'a23*b22',
                   'a31*b31', 'a32*b31', 'a33*b31', 'a31*b32', 'a32*b32', 'a33*b32',
                   'a41*b41', 'a42*b41', 'a42*b41', 'a41*b42', 'a42*b42', 'a42*b42']
    ref_fs_product = sympy_to_numpy(ref_symbols, (4, product_basis_size))

    # Loop-based
    z1 = iteration_implementation_face_splitting_product(a, b)
    assert z1.shape == (4, product_basis_size)
    assert np.array_equal(z1, ref_fs_product), "Expect same symbolic elements"

    # Single loop
    z2 = face_splitting_product_single_loop(a, b)
    assert z2.shape == (4, product_basis_size)
    assert np.array_equal(z2, ref_fs_product), "Expect same symbolic elements"

    # # No loops -> pure numpy
    z3 = face_splitting_product(a, b)
    assert z3.shape == (4, product_basis_size)
    assert np.array_equal(z3, ref_fs_product), "Expect same symbolic elements"


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

    assert wfs.shape == (1000, 22), "Expect (n_grid_points, n_occ_states)"
    assert rho.shape == (1000,), "Expect (n_grid_points)"
    assert grid.shape == (1000, 3), "Expect (n_grid_points, 3)"

    n_clusters = 100
    interp_points, indices = interpolation_points(grid, rho, n_clusters, random_state=10)
    print(interp_points.shape)
    print(len(indices))


    # assert np.allclose(grid[indices], interp_points)



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




# Should make some notes on this (broadcasting) in a jupyter NB
# def test_broadcasting():
#     a = np.array(sympy.symbols('a11 a12 '
#                                'a21 a22 '
#                                'a31 a32 '), object).reshape((3, 2))
#
#     b = np.array(sympy.symbols('b11 b12 '
#                                'b21 b22 '
#                                'b31 b32 '), object).reshape((3, 2))
#
#     c = a[:, :, np.newaxis] * b[:, np.newaxis, :]
#     print(c.reshape(3, -1))
