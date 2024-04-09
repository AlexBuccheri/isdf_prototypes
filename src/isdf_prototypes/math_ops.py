""" Maths operations

Functions
------------------

Face-splitting product:
 * `iteration_implementation_face_splitting_product`  For reference, and use in fortran
 * `face_splitting_product_single_loop`               For reference, and use in fortran
 * `face_splitting_product`                           For use in production
"""
from typing import Optional

import numpy as np


def iteration_implementation_face_splitting_product(phi: np.ndarray, psi: Optional[np.ndarray] = None) -> np.ndarray:
    """Face-splitting Product. Create products of KS wave functions on a real-space grid.

    Naive loop-based implementation as a reference.
    Lack of numpy use means no vectorisation (very slow).
    See [wikipedia](https://en.wikipedia.org/wiki/Khatri–Rao_product#Face-splitting_product) entry and associated
    image for a clear description of the algorithm:

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

    See [wikipedia](https://en.wikipedia.org/wiki/Khatri–Rao_product#Face-splitting_product) entry
    and associated image for a clear description of the algorithm:

    .. math::

        \mathbf{C} \bullet \mathbf{D} =
         \left[\begin{array}{l}
           \mathbf{C}_1 \otimes \mathbf{D}_1 \\
           \hline \mathbf{C}_2 \otimes \mathbf{D}_2 \\
           \hline \mathbf{C}_3 \otimes \mathbf{D}_3
         \end{array}\right]

    :param phi: KS states defined on real-space grid, expected shape(Ngrid, m) where m is the number of KS states.
    :param psi: An optional second set of KS states defined on real-space grid. If psi=None, it is assumed that {phi} = {psi}
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

    See [wikipedia](https://en.wikipedia.org/wiki/Khatri–Rao_product#Face-splitting_product) entry
    and associated image for a clear description of the algorithm:

    .. math::

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



# TODO Deal with this - already implemented once
# def find_subgrid_in_grid_single_loop(grid, subgrid) -> np.ndarray:
#     """
#
#     :return:
#     """
#     all_indices = []
#     for i in range(len(subgrid)):
#         indices = np.where(np.all(np.abs(grid - subgrid[i, :]) < 1.e-9,  axis=1))[0]
#         all_indices.append(indices)
#     indices = np.concatenate(all_indices)
#     return indices
#
#
# def test_find_subgrid_in_grid(benzene_dft):
#     _, _, _, grid = benzene_dft
#
#
#     # np.set_printoptions(precision=15)
#     # print(grid[9, :])
#     # print(grid[101, :])
#     # print(grid[8, :])
#     # print(grid[87, :])
#     # print(grid[999, :])
#
#     subgrid = np.array([[-7.626786546123851, -7.07273774366262 , 10.679261155133798],
#                         [-5.915393190274166, -7.07273774366262 ,  5.143903500910294],
#                         [-7.626786546123851, -7.07273774366262 ,  9.98734144835586 ],
#                         [-7.626786546123851,  5.500833471849858,  9.295421741577922],
#                         [ 7.775753656523316,  7.072529873788919, 10.679261155133798]])
#
#     indices = find_subgrid_in_grid_single_loop(grid, subgrid)
#     assert all(indices == [9, 101, 8, 87, 999])
#     assert np.allclose(grid[indices], subgrid)
#
