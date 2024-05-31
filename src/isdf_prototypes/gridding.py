"""

"""
import numpy as np


def indices_to_composite(indices, limits) -> int:
    """

    for ix in range(0, nx):
        for iy in range(0, ny):
            for iz in range(0, nz):


    :param indices: Loop indices, storing inner to outer from left to right
    :param limits: Limits of loops, storing inner to outer loop limits from left to right

    :return: icmp: Composite index a set of [iz, iy, ix, ...] indices of nested loops.
    """
    n_nested = len(indices)
    assert len(limits) == n_nested

    # Initialise
    inner = indices[0]
    outer = indices[1]
    icmp = inner + (outer * limits[0])

    # For 3 nested loops and more
    for i in range(2, n_nested):
        inner = icmp
        outer = indices[i]
        icmp = inner + (outer * np.prod(limits[0:i]))

    return icmp


def expand_index_to_two_indices(ixy, limits) -> tuple:
    """
    This takes limits = [ny, nx]
    and returns [iy, ix] for loop structure:

        for ix in range(nx):
            for iy in range(ny):

    :param limits: Limits of loops, storing inner to outer loop limits from left to right
    :return:
    """
    iy = ixy % limits[0]
    ix = int((ixy - iy) / limits[0])
    return iy, ix


def expand_index_to_three_indices(ixyz, limits) -> tuple:
    """

    This takes limits = [nz, ny, nx]
    and returns [iz, iy, ix] for loop structure:

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):

    :param limits: Limits of loops, storing inner to outer loop limits from left to right
    TODO Alex Might want to reverse this. See notes below
    :return:
    """
    iyz = ixyz % (limits[0] * limits[1])
    ix = int((ixyz - iyz) / (limits[0] * limits[1]))
    iz = iyz % limits[0]
    iy = int((iyz - iz) / limits[0])
    return iz, iy, ix


# Rename unroll
def expand_index_to_indices(icmp, limits) -> list:
    """

    This takes limits = [nz, ny, nx]
    and returns [iz, iy, ix] for loop structure:

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):

    TODO Alex Might want to reverse order that limits
    are specified in and corresponding, order that indices
    are returned. Would do so like:

    Use in initialisation:
      np.prod(limits[-2:]))
      indices = [outer]
    In loop:
       np.prod(limits[-i:]))
       indices.append(outer)
    indices.append(inner

    This takes limits = [nx, ny, nz]
    and returns [ix, iy, iz] for loop structure:

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz)

    :return:
    """
    n_nested = len(limits)
    indices = np.empty(shape=n_nested, dtype=np.int32)

    inner = icmp % np.prod(limits[:-1])
    outer = int((icmp - inner) / np.prod(limits[:-1]))
    indices[-1] = outer

    for i in range(2, n_nested):
        last_inner = inner
        inner = last_inner % np.prod(limits[:-i])
        outer = int((last_inner - inner) / np.prod(limits[:-i]))
        indices[-i] = outer
    indices[0] = inner

    return indices.tolist()
