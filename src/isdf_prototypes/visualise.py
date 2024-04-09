""" IO and visualisation tools
"""
from rdkit import Chem


def visualise_mo(cube_data, view, isoval, mol=None):
    """ Visualise cube file data with py3Dmol

    :param view: py3Dmol view instance
    :param file: Cube file
    :param mol: RDKIT molecule
    """
    # Isosurface
    view.addVolumetricData(cube_data, "cube", {'isoval': -isoval, 'color': "red", 'opacity': 0.75})
    view.addVolumetricData(cube_data, "cube", {'isoval': isoval, 'color': "blue", 'opacity': 0.75})

    # Ball and stick
    if mol:
        view.addModel(Chem.MolToMolBlock(mol), 'mol')
        view.setStyle({'stick': {}})

    view.zoomTo()
    # Running .update results in plot duplication
    # view.update()
    return view


def write_xyz(species, positions):
    """ Write xyz file for finite systems.

    :param species:
    :param positions:
    :return:
    """
    xyz_string = "\n"
    xyz_string += f'{positions.shape[0]}\n'
    for i in range(0, positions.shape[0]):
        xyz_string += f'{species[i]}   {positions[i, 0]:12.6f}   {positions[i, 1]:12.6f}   {positions[i, 2]:12.6f}\n'
    return xyz_string
