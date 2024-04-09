""" Functions to compute the ISDF vectors

 TODO
* Construction of P^phi for two sets of points (RS grid} and {Interp grid}
    * This trivially extends to the construction of  P^phi for two sets of points (Interp grid} and {Interp grid}
    * This trivially extends to the construction of of P^psi for the same grids

* Construction of Theta in terms of P matrices
    * If {phi} == {psi}, test construction of Theta in terms of Phi

* Comparison of KS product states to ISDF product states
    LHS:
        * Use face-splitting product for the L.H.S, with states evaluated on the whole grid
    RHS:
        * Use face-splitting product for the R.H.S, with states evaluated on the interpolation grid
        * Matrix-matrix multiplication with of face-splitting product on interp grid, with Theta
    Comparison of LHS with RHS.
        * Visually with cube files
        * Numerically: Avg, Min and Max Diff of LHS vs RHS
    * Should compare the scaling/flops of building LHS verse RHS
"""
from typing import Optional

import numpy as np


def construct_interpolation_vectors_matrix(phi: np.ndarray,
                                           interpolation_indices: np.ndarray,
                                           psi: Optional[np.ndarray] = None) -> np.ndarray:
    r""" Construct interpolation vectors matrix, \(\Theta\).

    .. math::

     \Theta &= \left[ Z C^T \right] \odot \left[ C C^T \right]^{-1} \\
            &= P^\Phi \odot P^\Psi \odot  \left[ \left[ P^\Phi \right]^\prime \odot  \left[P^\Psi \right]^\prime  \right]^{-1}

    :param phi: KS states defined for all points on the real-space grid.
                phi must have the shape (N_grid_points, m KS states)
    :param interpolation_indices: Grid indices that define interpolation points
    :param psi: Optional second set of KS states, with shape (N_grid_points, n KS states).
                Note that m does not need to equal n.
    :return theta: Interpolation vector matrix, with shape(N_grid_points, N_interpolation_vectors)
    """
    if psi is None:
        # Shallow copy
        psi = phi

    assert phi.shape == psi.shape, "phi and psi must have the same shape"

    # Initialise with Theta = ZC^T
    theta = (phi @ phi[interpolation_indices, :].T) * (psi @ psi[interpolation_indices, :].T)

    cct = (phi[interpolation_indices, :] @ phi[interpolation_indices, :].T) * (
           psi[interpolation_indices, :] @ psi[interpolation_indices, :].T)

    # Theta = ZC^T * [CC^T]^{-1}
    theta *= np.linalg.inv(cct)

    n_grid_points = phi.shape[0]
    assert theta.shape == (n_grid_points, len(interpolation_indices))

    return theta
