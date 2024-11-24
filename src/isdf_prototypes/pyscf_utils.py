""" Helper functions built using PYSCF
"""
import numpy as np

from pyscf.tools.cubegen import Cube
from pyscf.lib.misc import prange
from pyscf.dft import numint


def set_cube_defaults(resolution=None, margin=None):
    """

    :param resolution:
    :param margin:
    :return:
    """
    from pyscf import __config__

    # TODO Should check how these affect the sampling points on the grid
    # Default cube values
    if resolution is None:
        RESOLUTION = getattr(__config__, 'cubegen_resolution', None)
        resolution = RESOLUTION
    if margin is None:
        BOX_MARGIN = getattr(__config__, 'cubegen_box_margin', 3.0)
        margin = BOX_MARGIN
    return resolution, margin


def build_electron_density(mol, dm, nx=80, ny=80, nz=80, resolution=None, margin=None) -> np.ndarray:
    """ Build electron density from the density matrix

    :param mol: PYSCF Molecule
    :param dm:  Density matrix
    :param nx:  Sampling of grid in x dim
    :param ny:  Sampling of grid in y dim
    :param nz:  Sampling of grid in z dim
    :param resolution:
    :param margin:
    :return: Electron density as a flattened vector defined for all grid points
    """
    resolution, margin = set_cube_defaults(resolution, margin)

    # Hard-coded for not-periodic
    GTOval = 'GTOval'

    cc = Cube(mol, nx, ny, nz, resolution, margin)
    coords = cc.get_coords()

    # Total number of grid points
    ngrids = cc.get_ngrids()
    # Check what this is
    blksize = min(8000, ngrids)

    rho = np.empty(ngrids)
    for ip0, ip1 in prange(0, ngrids, blksize):
        ao = mol.eval_gto(GTOval, coords[ip0:ip1])
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)

    return rho


def build_mo_on_realspace_grid(mol, coeff, nx=80, ny=80, nz=80, resolution=None, margin=None) -> np.ndarray:
    """ Build a molecular orbital on a real space grid.

    Based on PYSCF cubegen.py/orbital routine. Hard-coded for molecular systems.

    One can reshape like orb_on_grid = orb_on_grid.reshape(cc.nx,cc.ny,cc.nz)
    which follows the grid convention:

    ```python
    for i in range(nx)
      for j in range(ny)
         for k in range(nz)
    ```
    :param mol: PYSCF Molecule
    :param coeff:  MO coefficents for a single MO state
    :param nx:  Sampling of grid in x dim
    :param ny:  Sampling of grid in y dim
    :param nz:  Sampling of grid in z dim
    :param resolution:
    :param margin:
    :return: MO wave function as a flattened vector defined for all grid points
    """
    resolution, margin = set_cube_defaults(resolution, margin)

    # Hard-coded for not-periodic
    GTOval = 'GTOval'

    cc = Cube(mol, nx, ny, nz, resolution, margin)
    coords = cc.get_coords()

    # Total number of grid points
    ngrids = cc.get_ngrids()
    # Check what this is
    blksize = min(8000, ngrids)

    # Flattened MO
    orb_on_grid = np.empty(ngrids)
    for ip0, ip1 in prange(0, ngrids, blksize):
        ao = mol.eval_gto(GTOval, coords[ip0:ip1])
        orb_on_grid[ip0:ip1] = np.dot(ao, coeff)

    return orb_on_grid
