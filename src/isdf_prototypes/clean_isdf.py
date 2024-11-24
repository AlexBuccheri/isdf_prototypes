""" Module meant to be used by the CLEAN_ISDF notebook
I'm just going to bang everything in here
"""
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from pyscf import gto, dft
from pyscf.tools import cubegen
from rdkit import Chem
import scipy
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from isdf_prototypes.helpers import pyscf_occupied_mos
from isdf_prototypes.math_ops import face_splitting_product


# Taken from https://docs.scipy.org/doc/scipy/reference/constants.html
bohr_to_ang = 0.529177210903


def benzene_from_pyscf(output_root: Path, points) -> dict:
    """ Compute the wave functions, real-space grid and density
    using PYSCF, for benzene.
    :return: Dict of the above
    """
    benzene_coordinates = """
     C         -0.65914       -1.21034        3.98683
     C          0.73798       -1.21034        4.02059
     C         -1.35771       -0.00006        3.96990
     C          1.43653       -0.00004        4.03741
     C         -0.65915        1.21024        3.98685
     C          0.73797        1.21024        4.02061
     H         -1.20447       -2.15520        3.97369
     H          1.28332       -2.15517        4.03382
     H         -2.44839       -0.00006        3.94342
     H          2.52722       -0.00004        4.06369
     H         -1.20448        2.15509        3.97373
     H          1.28330        2.15508        4.03386
    """

    # PYSCF Molecule
    mol = gto.Mole()
    mol.atom = benzene_coordinates
    mol.basis = 'def2-SVP'
    mol.build()

    # RDKIT molecule from SMILES data
    # rdkit_mol = Chem.MolFromSmiles("c1ccccc1")
    # rdkit_mol = Chem.AddHs(rdkit_mol)

    # Solve SCF for restricted KS-LDA
    mf = dft.RKS(mol)
    mf.kernel()
    n_occ = pyscf_occupied_mos(mf)
    assert n_occ == 21, "Occupied states for benzene, with this basis is 21"

    # Grid/cube settings
    n_total = np.prod(points)
    nx_ny_nz = dict(zip(['nx', 'ny', 'nz'], points))

    # Generate the real-space grid
    cube_grid = cubegen.Cube(mol, **nx_ny_nz)
    # grid_points = cube_grid.get_coords()
    # assert grid_points.shape == (n_total, 3)

    # Generate MOs on real-space grid
    wfs = np.empty(shape=(n_total, n_occ))
    for i in range(n_occ):
        cube_file = output_root / f'benzene_mo{i}.cube'
        # pyscf expects a string for fname
        molecular_orbital = cubegen.orbital(mol, cube_file.as_posix(), mf.mo_coeff[:, i], **nx_ny_nz)
        wfs[:, i] = molecular_orbital.reshape(-1)

    # Construct the density
    dm = mf.make_rdm1()
    cube_file = output_root / f'density.cube'
    rho = cubegen.density(mol, cube_file.as_posix(), dm, **nx_ny_nz)
    rho = rho.reshape(-1)

    return {'wfs': wfs, 'rho': rho, 'cube_grid': cube_grid}


def orthogonalised_gaussian_matrix(n_states: int, n_int: int, apply_checks=True):
    """ Generate a random, orthogonalised Gaussian matrix of shape(n_states, >=sqrt(n_int))

    :param n_states: Number of states (dim 1)
    :param n_int:  Number of interpolation vectors (rooted for dim 2)
    :param apply_checks: Check G is orthogonalised
    """
    p = int(np.ceil(np.sqrt(n_int)))

    # Create a Gaussian matrix
    rng = np.random.default_rng()
    gaussian = rng.standard_normal((n_states, p))

    # Orthogonalise the Gaussian matrix, with 'reduced' ensuring G has the shape of gaussian
    G, R = np.linalg.qr(gaussian, mode='reduced')
    assert G.shape == (n_states, p)

    # Confirm it's orthogonalised
    if apply_checks:
        GTG = G.T @ G
        assert GTG.shape == (p, p), "Expect a shape (p, p)"
        assert np.allclose(GTG, np.eye(p, p)), "Expect all columns of G to be orthonormal"

    return G


def subsampled_product_matrix(wfs, G):
    r""" Sub-sample the Z product matrix, such that N interpolation points
    can be found through QR decomposition.

    .. math::

    \tilde{Z}_{\alpha \beta}(\mathbf{r}) =
       \left(\sum_{i=1}^m \varphi_i(\mathbf{r}) G_{i \alpha}^{\varphi}\right)
       \left(\sum_{j=1}^n \psi_j(\mathbf{r}) G_{j \beta}^\psi\right)

    which I assume (THIS NEEDS CHECKING) reduces to:

    .. math::

    \tilde{Z}_{\alpha \beta}(\mathbf{r}) =
       \left(\sum_{i=1}^m \varphi_i(\mathbf{r}) G_{i \alpha}^{\varphi}\right)^2

    when one only has a single set of KS states.

    :param wfs: Wave functions, with shape(n grid points, n KS states)
    :param G: Orthogonalised Gaussian matrix, with shape(n KS states, p)
    :return: \tilde{Z}_{\alpha \beta}(\mathbf{r}) with shape(n grid points, p*p
    with p**2 >= N interpolation points.
    """
    n_states, p = G.shape
    n_total = wfs.shape[0]

    assert wfs.shape[1] == n_states, \
        "Second dimension of wave functions and first of Gaussian matrix should equal the number of states"

    z_alpha = wfs @ G
    assert z_alpha.shape == (n_total, p)

    z_tilde = face_splitting_product(z_alpha)
    assert z_tilde.shape == (n_total, int(p * p))

    return z_tilde


def sample_z_with_gaussian_matrix(phi, n_interp: int, psi: Optional = None, random_seed=None):
    """ Randomly sample the product matrix using Gaussian test matrices.

    More or less copy-pasted from the function randomly_sample_product_matrix in `interpolation_points_via_qr.py`
    Note that here, I use a different G matrix for sampling \p psi, regardless of whether psi = phi.
    This behaviour differs from `subsampled_product_matrix` above, and I am not sure which is correct.
    """
    if psi is None:
        # Shallow copy
        psi = phi

    assert psi.shape[0] == phi.shape[0], \
        "Both sets of wave functions should be defined for the same number of grid points"

    # m = number of KS states for {phi}
    n_grid_points, m = phi.shape
    # Number of KS states for {psi}
    n = psi.shape[1]
    p = int(np.ceil(np.sqrt(n_interp)))

    # Do not enforce orthogonality of columns for G1 or G2
    G1 = np.random.default_rng(seed=random_seed).normal(0.0, 1.0, size=(m, p))
    A = phi @ G1
    assert A.shape == (n_grid_points, p)

    G2 = np.random.default_rng(seed=random_seed).normal(0.0, 1.0, size=(n, p))
    B = psi @ G2
    assert B.shape == (n_grid_points, p)

    z_subspace = face_splitting_product(A, B)
    assert z_subspace.shape == (n_grid_points, p * p)
    return z_subspace


def interpolation_points_via_qrpivot(z_tilde: np.ndarray, n_int):
    """
    From z_tilde, perform Z^T P = QR, where the first n_int columns of P correspond
    to the indices of interpolation points
    :param z_tilde:
    :return:
    """
    n_total, p_squared = z_tilde.shape
    p = int(np.ceil(np.sqrt(n_int)))
    assert int(np.sqrt(p_squared)) == p, 'p sizing not consistent'

    R, pivot = scipy.linalg.qr(z_tilde.T, mode='r', pivoting=True)
    assert pivot.shape == (n_total,)

    # Interpolation points are given by the first n_int values of pivot
    indices = np.sort(pivot[0:n_int])
    return indices


def construct_interpolation_vectors_more_parts(phi, indices):
    npoints, n_states = phi.shape
    n_int = indices.size

    # Output me
    phi_mu = phi[indices, :]
    assert phi_mu.shape == (n_int, n_states)

    # Output me
    P_r_mu = phi @ phi_mu.T
    assert P_r_mu.shape == (npoints, n_int)

    # Output me
    zct = P_r_mu * P_r_mu

    # Output me
    P_mu_nu = phi_mu @ phi_mu.T
    assert P_mu_nu.shape == (n_int, n_int)
    assert scipy.linalg.issymmetric(P_mu_nu), 'P_mu_nu should be symmetric'

    # Output me
    cct = P_mu_nu * P_mu_nu
    assert np.allclose(cct, zct[indices, :])
    assert scipy.linalg.issymmetric(cct, rtol=1.e-4), 'CC^T should be symmetric'

    # Output me
    inversion = np.linalg.pinv  # np.linalg.inv or np.linalg.pinv
    inv_cct = inversion(cct)
    #assert scipy.linalg.issymmetric(inv_cct, rtol=1.e-4), '[CC^T]^-1 should be symmetric'

    isdf_vectors = zct @ inv_cct
    assert isdf_vectors.shape == (npoints, n_int)

    return isdf_vectors


def find_interpolation_points_factory(interpolation_method: str) -> Callable:
    """Factory function to return `find_interpolation_points`
    """
    print(f'Interpolation Method is {interpolation_method}')

    if interpolation_method == 'kmeans':

        def find_interpolation_points(n_int, **kwargs):
            grid_points = kwargs['grid_points']
            rho = kwargs['rho']

            k_means = KMeans(n_clusters=n_int, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0,
                             algorithm='lloyd')
            k_means.fit(grid_points, sample_weight=rho)

            clusters = k_means.cluster_centers_
            assert clusters.shape == (n_int, 3)

            # Discretise the continuous cluster points to the grid
            distances = cdist(clusters, grid_points)
            indices = np.argmin(distances, axis=1)
            return indices

    elif interpolation_method == 'orthogonalised_sampling':

        def find_interpolation_points(n_int: int, **kwargs):
            wfs = kwargs['wfs']
            n_states = wfs.shape[1]
            G = orthogonalised_gaussian_matrix(n_states, n_int)
            z_tilde = subsampled_product_matrix(wfs, G)
            indices = interpolation_points_via_qrpivot(z_tilde, n_int)
            return indices

    elif interpolation_method == 'nonorthogonalised_sampling':

        def find_interpolation_points(n_int: int, **kwargs):
            wfs = kwargs['wfs']
            z_tilde = sample_z_with_gaussian_matrix(wfs, n_int, random_seed=42)
            indices = interpolation_points_via_qrpivot(z_tilde, n_int)
            return indices

    else:
        raise ValueError(f'Erroneous Input Method: {interpolation_method}')

    return find_interpolation_points


def approximate_product_basis(phi, indices, isdf_vectors):
    n_int = indices.size
    n_points, n_states = phi.shape
    n_products = n_states ** 2
    assert isdf_vectors.shape == (n_points, n_int)

    # Product basis defined on the interpolation grid
    product_mu = face_splitting_product(phi[indices, :])
    assert product_mu.shape == (n_int, n_products)

    # Approx product expansion in ISDF vectors
    approx_product = isdf_vectors @ product_mu
    assert approx_product.shape == (n_points, n_products)

    return approx_product


def error_l2(f1, f2, dV):
    # TODO Alex. I'm pretty sure I have the error term wrong in fortran
    # But it does not explain why the plots look shit. Should try them for 100 - 200 centroids

    assert dV > 0.0, "Must have a finite volume element"
    diff = f1 - f2
    # Integrate over grid points
    err_ij = np.sum(diff * diff, axis=0) * dV

    err_ij = np.sqrt(err_ij)
    err = {'min': np.amin(err_ij), 'max': np.amax(err_ij), 'mean': np.mean(err_ij)}
    return err


def mean_norm(f1, dV):
    """
    :param f1:  Target function
    :param f2:  Approximate function
    :param grid: system grid
    """
    # Volume element required in discretisation of <f1, f1>
    norm_f1 = np.sqrt(dV * np.sum(f1**2, axis=0))
    mean_norm_f1 = np.mean(norm_f1)
    return mean_norm_f1
