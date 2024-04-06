""" Functions to compute the ISDF vectors

Functions
------------------

Face-splitting product:
 * `iteration_implementation_face_splitting_product`  For reference, and use in fortran
 * `face_splitting_product_single_loop`               For reference, and use in fortran
 * `face_splitting_product`                           For use in production

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


def iteration_implementation_face_splitting_product(phi: np.ndarray, psi: Optional[np.ndarray] = None) -> np.ndarray:
    """Face-splitting Product. Create products of KS wave functions on a real-space grid.

    Naive loop-based implementation as a reference.
    Lack of numpy use means no vectorisation (very slow).
    See wikipedia entry and associated image for a clear description of the algorithm:
    https://en.wikipedia.org/wiki/Khatri–Rao_product#Face-splitting_product

    :param phi: KS states defined on real-space grid, expected shape(Ngrid, m)
    where m is the number of KS states.
    :param psi: An optional second set of KS states defined on real-space grid.
    If psi=None, it is assumed that {phi} = {psi}
    :return: Z of shape(Ngrid, len(psi) * len(phi))
    """
    if psi is None:
        # Shallow copy
        psi = phi

    assert psi.shape[0] == phi.shape[0], ("Both sets of wave functions should be defined for the same number of grid "
                                          "points")
    n, m = psi.shape[1], phi.shape[1]
    n_grid = psi.shape[0]
    z = np.empty(shape=(n_grid, n * m), dtype=type(psi))

    for ir in range(n_grid):
        ij = 0
        for i in range(n):
            psi_element = psi[ir, i]
            for j in range(m):
                z[ir, ij] = psi_element * phi[ir, j]
                ij += 1
    return z


def face_splitting_product_single_loop(phi: np.ndarray, psi: Optional[np.ndarray] = None) -> np.ndarray:
    r""" Face-splitting Product. Create products of KS wave functions on a real-space grid.

    See wikipedia entry and associated image for a clear description of the algorithm:
    https://en.wikipedia.org/wiki/Khatri–Rao_product#Face-splitting_product

    ::math

        \mathbf{C} \bullet \mathbf{D} =
         \left[\begin{array}{l}
           \mathbf{C}_1 \otimes \mathbf{D}_1 \\
           \hline \mathbf{C}_2 \otimes \mathbf{D}_2 \\
           \hline \mathbf{C}_3 \otimes \mathbf{D}_3
         \end{array}\right]

    :param phi: KS states defined on real-space grid, expected shape(Ngrid, m)
    where m is the number of KS states.
    :param psi: An optional second set of KS states defined on real-space grid.
    If psi=None, it is assumed that {phi} = {psi}
    :return: Z of shape(Ngrid, len(psi) * len(phi))
    """
    if psi is None:
        # Shallow copy
        psi = phi

    assert psi.shape[0] == phi.shape[0], ("Both sets of wave functions should be defined for the same number of grid "
                                          "points")
    n, m = psi.shape[1], phi.shape[1]
    n_grid = psi.shape[0]
    z = np.empty(shape=(n_grid, n * m), dtype=type(psi))

    for ir in range(n_grid):
        z[ir, :] = np.kron(psi[ir, :], phi[ir, :])

    return z


def face_splitting_product(phi: np.ndarray, psi: Optional[np.ndarray] = None) -> np.ndarray:
    r"""Face-splitting Product. Create products of KS wave functions on a real-space grid.

    See wikipedia entry and associated image for a clear description of the algorithm:
    https://en.wikipedia.org/wiki/Khatri–Rao_product#Face-splitting_product

    ::math

        \mathbf{C} \bullet \mathbf{D} =
         \left[\begin{array}{l}
           \mathbf{C}_1 \otimes \mathbf{D}_1 \\
           \hline \mathbf{C}_2 \otimes \mathbf{D}_2 \\
           \hline \mathbf{C}_3 \otimes \mathbf{D}_3
         \end{array}\right]


    Fully vectorised implementation, with no python loops.
    For additional details on how to implement the full Khatri-Rao Product, see
    https://github.com/numpy/numpy/issues/15298

    :param phi: KS states defined on real-space grid, expected shape(Ngrid, m)
    where m is the number of KS states.
    :param psi: An optional second set of KS states defined on real-space grid.
    If psi=None, it is assumed that {phi} = {psi}
    :return: Z of shape(Ngrid, len(psi) * len(phi))
    """
    if psi is None:
        # Shallow copy
        psi = phi

    assert psi.shape[0] == phi.shape[0], ("Both sets of wave functions should be defined for the same number of grid "
                                          "points")
    n_grid = psi.shape[0]
    z = psi[:, :, np.newaxis] * phi[:, np.newaxis, :]
    z = z.reshape(n_grid, -1)
    return z


