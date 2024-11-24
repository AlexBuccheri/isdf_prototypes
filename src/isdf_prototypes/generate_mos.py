"""Use PYSCF to generate MOs on a real-space grid

python src/isdf_prototypes/generate_mos.py
"""
import numpy as np

from joblib import Memory
from pyscf import gto, dft, scf
from pyscf.tools import cubegen

from isdf_prototypes.pyscf_utils import build_electron_density
from isdf_prototypes.interpolation_points import weighted_kmeans


# PYSCF probably running threaded
# CacheWarning: Unable to cache to disk. Possibly a race condition in the creation of the directory.
# Exception: cannot pickle '_io.TextIOWrapper' object.
# Specify the directory for caching
# cache_dir = 'cache_dir'
# # Create a Memory object with disk-based caching
# mem = Memory(location=cache_dir, verbose=0)


def build_benzene() -> gto.Mole:
    """ Build Benzene molecule
    :return:
    """
    benzene_coordinates = '''
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
    '''
    mol = gto.Mole()
    mol.atom = benzene_coordinates
    mol.basis = 'def2-SVP'
    mol.build()
    return mol


def solve_scf(mol):
    """ Solve molecule using DFT (LDA is default)

    Caching the result as I'm using the same molecule for testing.
    HF would use mf = scf.RHF(mol)

    :param mol:
    :return:
    """
    # mf -> Mean field
    mf = dft.RKS(mol)
    mf.kernel()
    return mf


def write_initial_centroid_indices(n_total: int, n_centroids: int):
    """

    :return:
    """
    # Randomly sample N grid points as initial centroids
    with open('centroid_indices.dat', 'w') as fid:
        indices = np.random.choice(n_total, n_centroids, replace=False)
        fid.write(f"Centroid grid indices for grid of size {n_total, 3}")
        indices.tofile(fid, sep='\n')


def write_xyz(species, positions):
    """

    :param species:
    :param positions:
    :return:
    """
    xyz_string = f'{positions.shape[0]}\n'
    for i in range(0, positions.shape[0]):
        xyz_string += f'{species[i]}   {positions[i, 0]:12.6f}   {positions[i, 1]:12.6f}   {positions[i, 2]:12.6f}\n'
    return xyz_string


# def output_cube_minus_volumetric(n_points, spacing, positions, origin):
#     """
#
#     :param n_points:
#     :param sampling:
#     :param positions:
#     :return:
#     """
#     angstrom_to_bohr = 1.8897261254578281
#     nx, ny, nz = n_points
#     n_centroids = positions.shape[0]
#     spacing *= angstrom_to_bohr
#
#     cube_str = f"""Cube file for Interpolation Grid Points
# (represented as atoms - no volumetric data)
# {n_centroids:5}{"".join(f"{s:12.6f}" for s in origin * angstrom_to_bohr)}
# {nx:5d}{spacing[0, 0]:12.6f}{spacing[0, 1]:12.6f}{spacing[0, 2]:12.6f}
# {ny:5d}{spacing[1, 0]:12.6f}{spacing[1, 1]:12.6f}{spacing[1, 2]:12.6f}
# {nz:5d}{spacing[2, 0]:12.6f}{spacing[2, 1]:12.6f}{spacing[2, 2]:12.6f}
# """
#     # Atomic numbers and positions
#     # In this case, a dummy AN and the interpolation point vectors
#     for i in range(n_centroids):
#         pos_str = " ".join(f"{x:10.6f}" for x in positions[i, :] * angstrom_to_bohr)
#         cube_str += f'100   0.000000 {pos_str}\n'
#
#     return cube_str



# APPROACH 1
# # Load the points back in
# centroids = np.loadtxt("interpol_points_benzene.dat", skiprows=1)
#
# # Grid
# mol = build_benzene()
# default_res = None
# default_margin = 3.0
# nx, ny, nz = 10, 10, 10
# n_points = [nx, ny, nz]
#
# cube_grid = cubegen.Cube(mol, nx, ny, nz, default_res, default_margin)
#
# # These are integers, so convert to Cartesian
# dx = cube_grid.xs[-1] if len(cube_grid.xs) == 1 else cube_grid.xs[1]
# dy = cube_grid.ys[-1] if len(cube_grid.ys) == 1 else cube_grid.ys[1]
# dz = cube_grid.zs[-1] if len(cube_grid.zs) == 1 else cube_grid.zs[1]
# spacing = (cube_grid.box.T * [dx, dy, dz]).T
#
# cube_str = output_cube_minus_volumetric(n_points, spacing, centroids, cube_grid.boxorig)
# with open("centroids.cube", "w") as fid:
#     fid.write(cube_str)


# Approach 2
# Add centroid points as atoms to mol
# Output density (rho) to cube, using this molecule
# not sure this will work because it's going to loop over all atoms when generating the density

# Approach 3
# Output grid points as .xyz
# Overlay the density ISOSURFACE in VMD


mol = build_benzene()
mf = solve_scf(mol)
# 1-electron Density matrix
dm = mf.make_rdm1()

# Build the electron density
default_res = None
default_margin = 3.0
nx, ny, nz = 10, 10, 10

n_total = np.prod([nx, ny, nz])

rho = build_electron_density(mol, dm, nx=nx, ny=ny, nz=nz, resolution=default_res, margin=default_margin)
assert rho.shape == (n_total,)

# Export MOs in cube format
# 1st MO. 22nd MO (LUMO) = mf.mo_coeff[:, 21]
# cubegen.orbital(mol, 'benzene_lumo.cube', mf.mo_coeff[:, 21])

# Real-space grid
# Note, the same grid is built by calls to construct the density, and MOs.
cube_grid = cubegen.Cube(mol, nx, ny, nz, default_res, default_margin)
grid_points = cube_grid.get_coords()
assert grid_points.shape == (n_total, 3)

# Initialise centroid choice (cached from random choice)
n_centroids = 99
indices = np.loadtxt('centroid_indices.dat', skiprows=1, dtype=int)
assert len(indices) == n_centroids, "Number of loaded centroid indices inconsistent with expected amount"
initial_centroids = grid_points[indices]

# Find optimal centroid positions
# For fixed initial guess, this is deterministic
recompute_centroids = True
if recompute_centroids:
    max_iter = 100
    centroids, iter = weighted_kmeans(grid_points, rho, initial_centroids,
                                      n_iter=max_iter, centroid_tol=1.0e-9,
                                      safe_mode=True, verbose=False)
    if iter == max_iter:
        print("Max iterations was reached when finding optimal interpolation sampling points")

    # print("Optimal interpolation sampling points written to file")
    # with open('interpol_points_benzene.dat', 'w') as fid:
    #     header = f"# {n_centroids} optimal interpolation sampling points for benzene grid of size {n_total}"
    #     np.savetxt(fid, centroids, header=header)
else:
    # Load centroids points
    centroids = np.loadtxt("interpol_points_benzene.dat", skiprows=1, dtype=float)

grid_xyz = write_xyz(['U']*grid_points.shape[0],  grid_points)

with open("grid.xyz", "w") as fid:
    fid.write(grid_xyz)


centroids_xyz = write_xyz(['U']*centroids.shape[0], centroids)

with open("centroids.xyz", "w") as fid:
    fid.write(centroids_xyz)

