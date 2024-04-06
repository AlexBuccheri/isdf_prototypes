from typing import List

import numpy as np
import sympy

from isdf_prototypes.isdf_vectors import iteration_implementation_face_splitting_product, face_splitting_product_single_loop, face_splitting_product


# Helper function
def sympy_to_numpy(str_list: List[str], shape) -> np.ndarray:
    """ Convert a list of strings into a symbolic numpy array, with correct shape.
    :param str_list:
    :param shape:
    :return:
    """
    assert len(str_list) == np.prod(shape)
    array_sympy = [sympy.sympify(expr) for expr in str_list]
    array = np.array(array_sympy, dtype=object).reshape(shape)
    return array


def test_face_splitting_product():
    # Input array
    a = np.array(sympy.symbols('a11 a12 a13 '
                               'a21 a22 a23 '
                               'a31 a32 a33 '
                               'a41 a42 a42 '), object).reshape((4, 3))

    # Face-splitting product of a x a
    ref_symbols = ['a11**2', 'a11*a12', 'a11*a13', 'a11*a12', 'a12**2', 'a12*a13', 'a11*a13', 'a12*a13', 'a13**2',
                   'a21**2', 'a21*a22', 'a21*a23', 'a21*a22', 'a22**2', 'a22*a23', 'a21*a23', 'a22*a23', 'a23**2',
                   'a31**2', 'a31*a32', 'a31*a33', 'a31*a32', 'a32**2', 'a32*a33', 'a31*a33', 'a32*a33', 'a33**2',
                   'a41**2', 'a41*a42', 'a41*a42', 'a41*a42', 'a42**2', 'a42**2 ', 'a41*a42', 'a42**2 ', 'a42**2']
    ref_fs_product = sympy_to_numpy(ref_symbols, (4, 9))

    # Loop-based
    z1 = iteration_implementation_face_splitting_product(a, a)
    assert z1.shape == (4, 9)
    assert np.array_equal(z1, ref_fs_product), "Expect same symbolic elements"

    # Single loop
    z2 = face_splitting_product_single_loop(a, a)
    assert z2.shape == (4, 9)
    assert np.array_equal(z2, ref_fs_product), "Expect same symbolic elements"

    # No loops -> pure numpy
    z3 = face_splitting_product(a, a)
    assert z3.shape == (4, 9)
    assert np.array_equal(z3, ref_fs_product), "Expect same symbolic elements"


def test_face_splitting_product_with_different_num_states():

    a = np.array(sympy.symbols('a11 a12 a13 '
                               'a21 a22 a23 '
                               'a31 a32 a33 '
                               'a41 a42 a42 '), object).reshape((4, 3))

    b = np.array(sympy.symbols('b11 b12 '
                               'b21 b22 '
                               'b31 b32 '
                               'b41 b42 '), object).reshape((4, 2))

    assert a.shape[0] == b.shape[0], "a and b must have the same number of rows"
    product_basis_size = 6
    assert a.shape[1] * b.shape[1] == product_basis_size

    # Face-splitting product of a x b
    ref_symbols = ['a11*b11', 'a12*b11', 'a13*b11', 'a11*b12', 'a12*b12', 'a13*b12',
                   'a21*b21', 'a22*b21', 'a23*b21', 'a21*b22', 'a22*b22', 'a23*b22',
                   'a31*b31', 'a32*b31', 'a33*b31', 'a31*b32', 'a32*b32', 'a33*b32',
                   'a41*b41', 'a42*b41', 'a42*b41', 'a41*b42', 'a42*b42', 'a42*b42']
    ref_fs_product = sympy_to_numpy(ref_symbols, (4, product_basis_size))

    # Loop-based
    z1 = iteration_implementation_face_splitting_product(a, b)
    assert z1.shape == (4, product_basis_size)
    assert np.array_equal(z1, ref_fs_product), "Expect same symbolic elements"

    # Single loop
    z2 = face_splitting_product_single_loop(a, b)
    assert z2.shape == (4, product_basis_size)
    assert np.array_equal(z2, ref_fs_product), "Expect same symbolic elements"

    # # No loops -> pure numpy
    z3 = face_splitting_product(a, b)
    assert z3.shape == (4, product_basis_size)
    assert np.array_equal(z3, ref_fs_product), "Expect same symbolic elements"


# Should make some notes on this (broadcasting) in a jupyter NB
# def test_broadcasting():
#     a = np.array(sympy.symbols('a11 a12 '
#                                'a21 a22 '
#                                'a31 a32 '), object).reshape((3, 2))
#
#     b = np.array(sympy.symbols('b11 b12 '
#                                'b21 b22 '
#                                'b31 b32 '), object).reshape((3, 2))
#
#     c = a[:, :, np.newaxis] * b[:, np.newaxis, :]
#     print(c.reshape(3, -1))
