"""Helper functions for use in Jupyter
"""
from pathlib import Path
from string import Template
from typing import Tuple

import numpy as np
from scipy.constants import physical_constants

from pyscf.tools import cubegen
from pyscf import gto
from pyscf.gto.mole import Mole
from rdkit import Chem
from sklearn.cluster import KMeans

from isdf_prototypes.isdf_vectors import construct_interpolation_vectors_matrix
from isdf_prototypes.math_ops import face_splitting_product


bohr_to_ang = physical_constants["Bohr radius"][0] * 1.e10


# TODO rdkit_molecule does not work
# def rdkit_molecule(xyz_file):
#     """ Generate RDKIT ROMol from xyz file
#     Does not work
#     :param xyz_file:
#     :return:
#     """
#     from isdf_prototypes.xyz2mol import read_xyz_file
#     if not Path(xyz_file).is_file():
#         raise FileNotFoundError(f'Cannot find file {xyz_file}')
#
#     raw_mol = read_xyz_file(xyz_file)
#     print(raw_mol)
#     rdkit_mol = Chem.Mol(raw_mol)
#     # add hydrogens?
#     rdkit_mol = Chem.AddHs(rdkit_mol)
#     return rdkit_mol


def mean_squared_error_regular_grid(x, y):
    """ Mean squared error on a regular grid
    if every volume element is the same, i.e. a regular grid

     (dv / V) \sum^n_i (xi - y_i)^2

     where V = (n* dV), reduces to:

    (1 / n) \sum^n_i (xi - y_i)^2

    :return:
    """
    return np.mean((x - y) * (x - y))


def relative_error(x_ref, x) -> float:
    """
    Note, if the absolute value of the reference is small,
    small errors will blow up.
    :param x_ref:
    :param x:
    :return:
    """
    # Avoid division by zero
    mask = x_ref != 0
    error_rel = np.abs((x_ref[mask] - x[mask]) / x_ref[mask])
    mean_error_rel = np.mean(error_rel)
    return mean_error_rel


def duplicate_indices(a: np.ndarray, dim=0) -> list:
    """ Return the indices of any duplicate vectors in
    a numpy array
    """
    n_vectors = a.shape[dim]
    all_indices = np.arange(dim, n_vectors)
    _, unique_indices = np.unique(a, return_index=True, axis=dim)
    dup_indices = set(all_indices) - set(unique_indices)
    return list(dup_indices)


def find_interpolation_points_via_kmeans(n_clusters, grid, weight, **kwargs):
    """ Find interpolation points using SKLearn's KMeans.

    :return:
    """
    kmeans_args = {'init': 'k-means++',
                   'max_iter': 300,
                   'tol': 0.0001,
                   'verbose': 0,
                   'copy_x': True,
                   'random_state': None,
                   'algorithm': 'lloyd'}

    # Replace defaults with any user-specified values
    for key in kmeans_args.keys():
        try:
            kmeans_args[key] = kwargs[key]
        except KeyError:
            pass

    # Initialise
    k_means = KMeans(n_clusters=n_clusters, **kwargs)

    # Assigning weight function: https://stackoverflow.com/questions/50789660/weighted-k-means-in-python
    weight = weight.reshape(-1)
    assert weight.shape[0] == grid.shape[0]

    # Fit with points and weights
    k_means.fit(grid, sample_weight=weight)

    # Retrieve and return continuous values
    clusters = k_means.cluster_centers_
    assert clusters.shape == (n_clusters, 3)

    return clusters


def discretise_continuous_values(continuous_values, discrete_values) -> Tuple[np.ndarray, np.ndarray]:
    """Assign a set of continuous values to the closest discrete points.
    """
    n_values, n_dim = continuous_values.shape
    assert discrete_values.shape[1] == n_dim, "Components per vector differ for continuous and discrete arrays"

    discretised_values = np.empty_like(continuous_values)
    interpolation_indices = np.empty(shape=n_values, dtype=np.int32)

    for i in range(n_values):
        diff = discrete_values - continuous_values[i]
        # Grid index of closest discrete point
        ir = np.argmin(np.linalg.norm(diff, axis=1))
        discretised_values[i, :] = discrete_values[ir, :]
        interpolation_indices[i] = ir

    return discretised_values, interpolation_indices


def smiles_to_rdkitmol(smiles_str: str):
    rdkit_mol = Chem.MolFromSmiles(smiles_str)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    return rdkit_mol


def pyscf_molecule(xyz_file, basis: str) -> Mole:
    """ Generate PYSCF molecule from .xyz file and basis

    :param xyz_file:
    :param basis:
    :return:
    """
    if not Path(xyz_file).is_file():
        raise FileNotFoundError(f'Cannot find file {xyz_file}')

    with open(file=xyz_file, mode='r') as fid:
        coordinates = fid.readlines()[2:]

    mol = gto.Mole()
    mol.atom = coordinates
    mol.basis = basis
    mol.build()

    return mol


def pyscf_density(mf, points, cube_file: str) -> np.ndarray:
    dm = mf.make_rdm1()
    nx, ny, nz = points
    rho = cubegen.density(mf.mol, cube_file, dm, nx=nx, ny=ny, nz=nz)
    assert rho.shape == (nx, ny, nz)
    return rho


def pyscf_grid(mol, points) -> np.ndarray:
    nx, ny, nz = points
    n_total = np.prod((points))
    cube_grid = cubegen.Cube(mol, nx, ny, nz)
    grid = cube_grid.get_coords()
    assert grid.shape == (n_total, 3)
    return grid


def pyscf_occupied_mos(mf, occ_tol=1.e-8) -> int:
    """ Find the number of occupied molecular orbitals

    :param mf: Mean field instance
    :param occ_tol: Tolerance above which an MO is considered occupied
    :return: Number of occupied MOs
    """
    occupied_mo_indices = np.argwhere(mf.mo_occ > occ_tol)
    n_occ = len(occupied_mo_indices)
    return n_occ


def pyscf_molecular_orbitals(mf, n_points: list, cube_file: Template) -> np.ndarray:
    """ Compute all occupied molecular orbitals.

    :param mf: Mean field instance
    :param n_points: List containing the number of grid points per dimension
    :param cube_file: File name of the form Template("some/path/2/benzene_$i.cube")
    :return: wfs of shape(n_total, n_occ)
    """
    # Wave function dimensions
    nx, ny, nz = n_points
    n_total = np.prod(n_points)
    n_occ = pyscf_occupied_mos(mf)
    wfs = np.empty(shape=(n_total, n_occ))

    for i in range(n_occ):
        file_name = cube_file.substitute(i=i)
        molecular_orbital = cubegen.orbital(mf.mol, file_name, mf.mo_coeff[:, i], nx=nx, ny=ny, nz=nz)
        wfs[:, i] = molecular_orbital.reshape(-1)

    return wfs


def add_grid_to_view(view, grid, **kwargs):
    """

    :return:
    """
    n_total = grid.shape[0]
    for i in range(n_total):
        x, y, z = grid[i] * bohr_to_ang
        view.addSphere({'center': {'x': x, 'y': y, 'z': z}} | kwargs)
    # view.setStyle({'sphere': {}})
    return view


def compute_approximate_product_matrix(wfs, interpolation_indices) -> np.ndarray:
    """ Compute an approximate product matrix using ISDF interpolation vectors

     z_{\text{isdf}} = \sum_\mu^{N_\mu} \zeta_{\mu}(\mathbf{r}) \phi_i(\mathbf{r}_\mu) \phi_j(\mathbf{r}_\mu)

    which can be represented as the matrix equation:

    Z_{\text{isdf}} = \Zeta (Phi \bullet \Psi),

    where (Phi \bullet \Psi) is the face-splitting product.
    """
    n_interp = interpolation_indices.shape[0]
    total_grid_size, n_states = wfs.shape
    n_products = n_states**2

    # Construct the interpolation points vectors
    zeta = construct_interpolation_vectors_matrix(wfs, interpolation_indices)
    assert zeta.shape == (total_grid_size, n_interp)

    # Product basis defined on the interpolation grid
    z_interp = face_splitting_product(wfs[interpolation_indices, :])
    assert z_interp.shape == (n_interp, n_products)

    # ISDF approximation to the product basis
    z_isdf = zeta @ z_interp
    assert z_isdf.shape == (total_grid_size, n_products)

    return z_isdf


def write_grid_func_to_binary(fname: str, data: np.ndarray, for_fortran=True):
    """ Write a function defined on a grid to binary

    Single line of header is in real-text, for pre-parsing and
    the rest of the file is a binary stream.

    Example Usage:
        write_grid_func_to_binary("grid.bin", grid_points)
        write_grid_func_to_binary("wfs.bin", wfs)
        write_grid_func_to_binary("density.bin", rho)

    :param data: Function defined on a discrete grid
    :param for_fortran: Optional, transposes the shape written to file
    such that the array parses correctly in fortran
    ```fortran
    read(unit, *) ndim, ntotal
    allocate(grid(ndim, ntotal))
    ```
    :return:
    """
    # Header defines the transpose of the shape
    if for_fortran:
        s = data.shape[::-1]
    else:
        s = data.shape

    header = " ".join(str(x) for x in s) + '\n'

    with open(fname, 'wb') as fid:
        fid.write(header.encode('ascii'))
        fid.write(data.tobytes())
