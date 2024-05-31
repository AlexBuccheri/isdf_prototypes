import numpy as np

from isdf_prototypes.gridding import (indices_to_composite, expand_index_to_two_indices,
                                      expand_index_to_three_indices, expand_index_to_indices)


def test_indices_to_composite():

    # Note, this could be refactored to generate permutations of indices with numpy
    # rather than looping

    # Two loops
    nx = 2
    ny = 3
    limits = [ny, nx]

    ref_index = np.empty(shape=np.prod(limits))
    index = np.empty_like(ref_index)

    ir = -1
    for ix in range(0, nx):
        for iy in range(0, ny):
            ir += 1
            ref_index[ir] = ir
            icmp = indices_to_composite([iy, ix], limits)
            index[ir] = icmp

    assert (index == ref_index).all()

    # Three loops
    nx = 2
    ny = 3
    nz = 4
    limits = [nz, ny, nx]

    ref_index = np.empty(shape=np.prod(limits))
    index = np.empty_like(ref_index)

    ir = -1
    for ix in range(0, nx):
        for iy in range(0, ny):
            for iz in range(0, nz):
                ir += 1
                ref_index[ir] = ir
                icmp = indices_to_composite([iz, iy, ix], limits)
                index[ir] = icmp

    assert (index == ref_index).all()

    # Four loops
    nx = 2
    ny = 3
    nz = 4
    nk = 5

    limits = [nk, nz, ny, nx]

    ref_index = np.empty(shape=np.prod(limits))
    index = np.empty_like(ref_index)

    ir = -1
    for ix in range(0, nx):
        for iy in range(0, ny):
            for iz in range(0, nz):
                for ik in range(0, nk):
                    ir += 1
                    ref_index[ir] = ir
                    icmp = indices_to_composite([ik, iz, iy, ix], limits)
                    index[ir] = icmp

    assert (index == ref_index).all()


def test_expand_index_to_indices():

    # Two loops
    nx = 3
    ny = 4
    limits = [ny, nx]

    ref_index = np.empty(shape=(np.prod(limits), 2))
    index = np.empty_like(ref_index)

    ir = -1
    for ix in range(0, nx):
        for iy in range(0, ny):
            ir += 1
            ref_index[ir, :] = [iy, ix]
            jy, jx = expand_index_to_two_indices(ir, limits)
            index[ir, :] = [jy, jx]

    assert (index == ref_index).all()

    # Three loops
    nx = 2
    ny = 3
    nz = 4
    limits = [nz, ny, nx]

    ref_index = np.empty(shape=(np.prod(limits), 3))
    index = np.empty_like(ref_index)

    ir = -1
    for ix in range(0, nx):
        for iy in range(0, ny):
            for iz in range(0, nz):
                ir += 1
                ref_index[ir, :] = [iz, iy, ix]
                jz, jy, jx = expand_index_to_three_indices(ir, limits)
                index[ir, :] = [jz, jy, jx]

    assert (index == ref_index).all()

    # Four loops
    nx = 2
    ny = 3
    nz = 4
    nk = 5
    limits = [nk, nz, ny, nx]

    ref_index = np.empty(shape=(np.prod(limits), 4))
    index = np.empty_like(ref_index)

    ir = -1
    for ix in range(0, nx):
        for iy in range(0, ny):
            for iz in range(0, nz):
                for ik in range(0, nk):
                    ir += 1
                    ref_index[ir, :] = [ik, iz, iy, ix]
                    jk, jz, jy, jx = expand_index_to_indices(ir, limits)
                    index[ir, :] = [jk, jz, jy, jx]

    assert (index == ref_index).all()