"""

"""
import abc
from pathlib import Path
from typing import List

import numpy as np
from pyscf import gto, dft
from pyscf.tools import cubegen

from . import tests_root


class PYSCFCalc(abc.ABC):
    """ PYSCF Data required to compute wave functions and density on a real-space grid

    Attributes include an instance of the molecule and mean-field calculation, to
    speed up testing.
    """
    name: str
    basis: int
    coordinates: str
    n_points: List[float]
    mol: any = None
    mf: any = None
    nocc: int

    def molecule(self):
        ...

    def run_dft(self):
        ...


class MinimalBasisBenzene(PYSCFCalc):
    """ Minimal-basis Benzene instance.
    """
    name = 'benzene'
    basis = 'def2-SVP'
    coordinates = """
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
    # Real-space grid points
    n_points = [10, 10, 10]
    # For benzene with 'def2-SVP' basis, expect 22 occupied states
    nocc = 22

    def molecule(self):
        # Build a molecule instance
        mol = gto.Mole()
        mol.atom = self.coordinates
        mol.basis = self.basis
        mol.build()
        self.mol = mol

    def run_dft(self):
        # Solve SCF for restricted KS-LDA
        mf = dft.RKS(self.mol)
        mf.kernel()
        self.mf = mf


def compute_real_space_grid(calc: PYSCFCalc):
    nx, ny, nz = calc.n_points
    n_total = np.prod(calc.n_points)
    cube_grid = cubegen.Cube(calc.mol, nx, ny, nz)
    grid = cube_grid.get_coords()
    assert grid.shape == (n_total, 3)
    return grid


def compute_real_space_density(calc: PYSCFCalc, cube_file: str):
    nx, ny, nz = calc.n_points
    # 1-electron Density matrix
    dm = calc.mf.make_rdm1()
    rho = cubegen.density(calc.mol, cube_file, dm, nx=nx, ny=ny, nz=nz)
    rho = rho.reshape(-1)
    return rho


def generate_ks_states(calc: PYSCFCalc) -> np.ndarray:
    """Compute occupied MOs on a real-space grid

    :param calc: Calculation specification
    :return wfs: Occupied wave functions of shape (n_points, n_occ)
    """
    nx, ny, nz = calc.n_points
    n_grid = np.prod(calc.n_points)
    n_occ = calc.nocc
    wfs = np.empty(shape=(n_grid, n_occ))

    # Put unnecessary cube files in their own subdir
    cube_root = tests_root / Path("cube_files")
    cube_root.mkdir(exist_ok=True)

    for i in range(n_occ):
        # pyscf expects a string
        cube_file = cube_root.as_posix() + f'/{calc.name}_mo{i}.cube'
        molecular_orbital = cubegen.orbital(calc.mol, cube_file, calc.mf.mo_coeff[:, i], nx=nx, ny=ny, nz=nz)
        wfs[:, i] = molecular_orbital.reshape(-1)

    return wfs


def find_subgrid_in_grid(grid, subgrid) -> np.ndarray:
    """

    :return:
    """
    return np.where(np.allclose(grid[:, None, :], subgrid).all(axis=-1))[1]


def find_subgrid_in_grid_single_loop(grid, subgrid) -> np.ndarray:
    """

    :return:
    """
    all_indices = []
    for i in range(len(subgrid)):
        indices = np.where(np.all(np.abs(grid - subgrid[i, :]) < 1.e-9,  axis=1))[0]
        all_indices.append(indices)
    indices = np.concatenate(all_indices)
    return indices
