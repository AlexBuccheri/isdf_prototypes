""" Interpolation points via QR decomposition, with column-pivoting
"""
from typing import Optional

import numpy as np
import scipy

from isdf_prototypes.math_ops import face_splitting_product




def randomly_sample_product_matrix(phi, n_interp: int, psi: Optional = None, random_seed=None):
    """ Randomly sample the product matrix using Gaussian test matrices.

    .. math:

      \tilde{Z}_{\alpha, \beta} = \left( \sum^m_{i=1} \phi_i(\mathbf{r}) G^{\phi}_{i, \alpha} \right)
                                  \left( \sum^n_{j=1} \psi_j(\mathbf{r}) G^{\psi}_{j, \beta} \right)

    Implemented according to eq. 20 of "Interpolative Separable Density Fitting Decomposition for
    Accelerating Hybrid Density Functional Calculations with Applications to Defects in Silicon"
    J. Chem. Theory Comput. 2017, 13, 5420-5431

    :param phi: Function of shape(N_grid_points, m)
    :param n_interp: Requested number of interpolation points. This function will round the number such that
           sqrt(n_interp) is an integer.
    :param psi: Optional function of shape(N_grid_points, n). If not supplied, {psi} = {phi}.
    :param random_seed: Optional random seed. To be supplied for testing, such that the Gaussian matrices produce
           the same distributions on each call. A number between 1 and 42 is recommended.
    :return: z_subspace: A subspace of the pair product basis, on the real-space grid, with shape(N_grid_points, n_interp),
             where n_interp is modified such that its root is an integer.
    """
    if psi is None:
        # Shallow copy
        psi = phi

    assert psi.shape[0] == phi.shape[0], ("Both sets of wave functions should be defined for the same number of grid "
                                          "points")
    # m = number of KS states for {phi}
    n_grid_points, m = phi.shape
    # Number of KS states for {psi}
    n = psi.shape[1]
    p = int(np.round(np.sqrt(n_interp)))

    # TODO(Alex) I forgot to enforce orthogonality of columns for G1 or G1
    G1 = np.random.default_rng(seed=random_seed).normal(0.0, 1.0, size=(m, p))
    A = phi @ G1
    assert A.shape == (n_grid_points, p)

    G2 = np.random.default_rng(seed=random_seed).normal(0.0, 1.0, size=(n, p))
    B = psi @ G2
    assert B.shape == (n_grid_points, p)

    z_subspace = face_splitting_product(A, B)
    assert z_subspace.shape == (n_grid_points, p * p)
    return z_subspace




# def randomly_sample_product_matrix(phi, n_interp: int, psi: Optional = None, random_seed=None):
#     """ Randomly sample the product matrix using Gaussian test matrices.
#
#     .. math:
#
#       \tilde{Z}_{\alpha, \beta} = \left( \sum^m_{i=1} \phi_i(\mathbf{r}) G^{\phi}_{i, \alpha} \right)
#                                   \left( \sum^n_{j=1} \psi_j(\mathbf{r}) G^{\psi}_{j, \beta} \right)
#
#     Implemented according to eq. 20 of "Interpolative Separable Density Fitting Decomposition for
#     Accelerating Hybrid Density Functional Calculations with Applications to Defects in Silicon"
#     J. Chem. Theory Comput. 2017, 13, 5420-5431
#
#     :param phi: Function of shape(N_grid_points, m)
#     :param n_interp: Requested number of interpolation points. This function will round the number such that
#            sqrt(n_interp) is an integer.
#     :param psi: Optional function of shape(N_grid_points, n). If not supplied, {psi} = {phi}.
#     :param random_seed: Optional random seed. To be supplied for testing, such that the Gaussian matrices produce
#            the same distributions on each call. A number between 1 and 42 is recommended.
#     :return: z_subspace: A subspace of the pair product basis, on the real-space grid, with shape(N_grid_points, n_interp),
#              where n_interp is modified such that its root is an integer.
#     """
#     if psi is None:
#         # Shallow copy
#         psi = phi
#
#     assert psi.shape[0] == phi.shape[0], ("Both sets of wave functions should be defined for the same number of grid "
#                                           "points")
#     # m = number of KS states for {phi}
#     n_grid_points, m = phi.shape
#     # Number of KS states for {psi}
#     n = psi.shape[1]
#     p = int(np.round(np.sqrt(n_interp)))
#
#     # TODO(Alex) I forgot to enforce orthogonality of columns for G1 or G1
#     G1 = np.random.default_rng(seed=random_seed).normal(0.0, 1.0, size=(m, p))
#     A = phi @ G1
#     assert A.shape == (n_grid_points, p)
#
#     G2 = np.random.default_rng(seed=random_seed).normal(0.0, 1.0, size=(n, p))
#     B = psi @ G2
#     assert B.shape == (n_grid_points, p)
#
#     z_subspace = face_splitting_product(A, B)
#     assert z_subspace.shape == (n_grid_points, p * p)
#     return z_subspace


def interpolation_points_via_qrpivot(z_subspace) -> np.ndarray:
    """ Find interpolation points from a subspace of the pair-product matrix,
    using QR decomposition with pivoting.

    For QR with pivoting:
    * https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html
    * https://www.quantstart.com/articles/QR-Decomposition-with-Python-and-NumPy/

    :param z_subspace: Sub-sampled pair-product matrix, with shape(n_grid_points, n_interp)
    :return: pivot: Sorted interpolation point indices. These determine the rows of
    the product-basis matrix, Z, to be used as interpolating points. Note, n_rows should
    be the same for Z and the z_subspace.
    """
    n_interp = z_subspace.shape[1]
    # Do not return Q
    mode = 'r'
    R_and_pivot_matrices = scipy.linalg.qr(z_subspace.T, mode=mode, pivoting=True)
    R, pivot = R_and_pivot_matrices
    # Interpolation points are given by the first n_interp values of pivot
    print('Sizes:', z_subspace.shape, pivot.shape, n_interp)
    pivot = pivot[0:n_interp]
    return np.sort(pivot)
