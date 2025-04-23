"""
A collection of functions which perform coordinate transformations
• rot2d(...) - two-dimensional rotation about the origin
• cart2pol(...) - cartesian to polar coordinates in common frame
• pol2cart(...) - polar to cartesian coordinates in common frame
• local_pol_2_global_cart(...) - polar coordinates in a local frame to cartesian coordinates in global frame
• global_cart_2_local_pol(...) - cartesian coordinates in a global frame to polar coordinates in a local frame
• circle_inv_cart(...) - inversion about a circle
• test case() - simple test of included functions
"""
import numpy as np

def _convert_to_mxn(xy: np.ndarray |
                     tuple[float, float] |
                     tuple[tuple[float, float], ...] |
                     tuple[list[float], ...] |
                     list[float] |
                     list[tuple[float, float]] |
                     list[list[float]],
                 dim_m: float) -> tuple[np.ndarray, bool, bool]:
    """
    Helper function which converts an input sequence or array to an array of m rows by n columns, where m is specified and n is arbitrary
    :param xy: a sequence or array
    :param dim_m: number of rows in output array
    :return: tuple containing the [mxn] array, flag indicating the input was a vector, flag indicating the input was transposed
    """
    # Convert to numpy array
    if not isinstance(xy, np.ndarray):
        xy = np.array(xy)

    # if xy is vector, add dimension
    vector_flag = False
    if len(xy.shape) == 1:
        vector_flag = True
        xy = xy[:, None]

    # Ensure array is correct shape for calculation [mxn]
    transpose_flag = False
    if xy.shape[0] != dim_m and xy.shape[1] == dim_m:
        transpose_flag = True
        xy = xy.T

    return xy, vector_flag, transpose_flag

def _match_shape_of_out_to_in(xy: np.ndarray,
                              vector_flag: bool = False,
                              transpose_flag: bool = True) -> np.ndarray:
    """
    Helper function to quickly transpose and reduce dimension of input when necessary
    :param xy: array of dimension [mxn] which was output from _convert_to_mxn(...)
    :param vector_flag: flag indicating if xy should be a vector, and therefore empty dimensions should be removed
    :param transpose_flag: flag indicating if the output should be of dimension [nxm]
    :return: reshaped array
    """
    # Transpose
    if transpose_flag:
        xy = xy.T

    # Reduce dimension in case of vector
    # Note: transpose_flag and vector_flag are mutually exclusive, but this is not explicitly stated.
    if vector_flag:
        xy = xy.T.squeeze()

    return xy

# ----------------------------------------------------------------------------------------------------------------------
# 2D Rotation
# ----------------------------------------------------------------------------------------------------------------------
def rot2d(xy: np.ndarray |
              tuple[float, float] |
              tuple[tuple[float, float], ...] |
              tuple[list[float], ...] |
              list[float] |
              list[tuple[float, float]] |
              list[list[float]],
          th: float |
              tuple[float, ...] |
              list[float] |
              np.ndarray,
          deg: bool = False) -> np.ndarray:
    """
    Rotate 2d cartesian coordinates about the origin
    :param xy: a sequence or array of cartesian coordinate pairs, i.e.
    • [2xn]: [[x0, x1, ..., xn-1], [y0, y1, ..., yn-1]]
    • [nx2]: [[x0, y0], [x1, y1], ..., [xn-1, yn-1]]
    If the shape of xy is [nx2], and n!=2, xy is transposed before calculation.
    If xy is a tuple or list, it is converted to a Numpy array.
    :param th: angle of rotation about the origin.
    If the argument is a scalar, rotation is applied to all coordinate pairs.
    If the argument is vector-like, rotation of the ith angle is applied to the ith pair, and it must contain exactly n values.
    :param deg: Set to True if unit of measure of input angle is degrees; otherwise unit of measure is radians
    :return: an array of n rotated cartesian coordinate pairs, with shape identical to the input.
    """

    # Convert input to dimension [2xn]
    xy, v_flag, t_flag = _convert_to_mxn(xy, 2)

    # Convert degrees to radians
    if deg:
        th = np.array(th)
        th = np.pi * th / 180

    # Rotation matrix
    r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=float)

    # Apply rotation for single angle
    if isinstance(th, float | int):
        xy = np.matmul(r, xy)
    # Apply rotation for n angles
    else:
        # r is [2x2xn] (ijk)
        # xy is [2xn] (jk)
        # perform standard matrix multiplication for ith r and xy
        # rotated xy is [2xn] (ik)
        xy = np.einsum('ijk,jk->ik', r, xy)

    # Match output shape to input shape
    xy = _match_shape_of_out_to_in(xy, v_flag, t_flag)

    return xy


# ----------------------------------------------------------------------------------------------------------------------
# Cartesian to Polar
# ----------------------------------------------------------------------------------------------------------------------
def cart2pol(xy: np.ndarray |
                 tuple[float, float] |
                 tuple[tuple[float, float], ...] |
                 tuple[list[float], ...] |
                 list[float] |
                 list[tuple[float, float]] |
                 list[list[float]]) -> np.ndarray:
    """
    Convert from cartesian coordinates to polar coordinates in a common frame
    :param xy: a sequence or array of cartesian coordinate pairs, i.e.
    • [2xn]: [[x0, x1, ..., xn-1], [y0, y1, ..., yn-1]]
    • [nx2]: [[x0, y0], [x1, y1], ..., [xn-1, yn-1]]
    If the shape of xy is [nx2], and n!=2, xy is transposed before calculation.
    If xy is a tuple or list, it is converted to a Numpy array.
    :return: an array of polar coordinate pairs [Θ, r], with shape identical to the input
    """

    # Convert input to dimension [2xn]
    xy, v_flag, t_flag = _convert_to_mxn(xy, 2)

    # Transform
    tr = np.array(
        (np.arctan2(xy[1, :], xy[0, :]),
         np.hypot(xy[0, :], xy[1, :])),
        dtype=float
    )

    # Match output shape to input shape
    tr = _match_shape_of_out_to_in(tr, v_flag, t_flag)

    return tr


# ----------------------------------------------------------------------------------------------------------------------
# Polar to Cartesian
# ----------------------------------------------------------------------------------------------------------------------
def pol2cart(tr: np.ndarray |
                 tuple[float, float] |
                 tuple[tuple[float, float], ...] |
                 tuple[list[float, float], ...] |
                 list[float] |
                 list[tuple[float, float]] |
                 list[list[float, float]]) -> np.ndarray:
    """
    Convert from polar coordinates to cartesian coordinates in a common frame
    :param tr: a sequence or array of polar coordinate pairs, i.e.
    • [2xn]: [[Θ0, Θ1, ..., Θn-1], [r0, r1, ..., rn-1]]
    • [nx2]: [[Θ0, r0], [Θ1, r1], ..., [Θn-1, rn-1]]
    If the shape of xy is [nx2], and n!=2, tr is transposed before calculation.
    If tr is a tuple or list, it is converted to a Numpy array.
    :return: an array of cartesian coordinate pairs [x, y], with shape identical to the input
    """
    # Convert input to dimension [2xn]
    tr, v_flag, t_flag = _convert_to_mxn(tr, 2)

    # Transform
    xy = tr[1, :] * np.array(
        (np.cos(tr[0, :]),
         np.sin(tr[0, :])),
        dtype=float
    )

    # Match output shape to input shape
    xy = _match_shape_of_out_to_in(xy, v_flag, t_flag)

    return xy


# ----------------------------------------------------------------------------------------------------------------------
# Convert data in global cartesian coordinates to local polar coordinates
# ----------------------------------------------------------------------------------------------------------------------
def global_cart_2_local_pol(xy: np.ndarray |
                                tuple[float, float] |
                                tuple[tuple[float, float], ...] |
                                tuple[list[float], ...] |
                                list[float] |
                                list[tuple[float, float]] |
                                list[list[float]],
                            qf: np.ndarray |
                                tuple[float, float, float] |
                                tuple[tuple[float, float, float], ...] |
                                tuple[list[float], ...] |
                                list[float] |
                                list[tuple[float, float, float]] |
                                list[list[float]]) -> np.ndarray:
    """
    Convert cartesian data in a global frame, to polar data in a local frame.
    :param xy: a sequence or array of cartesian coordinate pairs
    • [2xn]: [[x0, x1, ..., xn-1], [y0, y1, ..., yn-1]]
    • [nx2]: [[x0, y0], [x1, y1], ..., [xn-1, yn-1]]
    :param qf: local frame configuration in the global frame
    • [3x1]: [x, y, φ]
    or a sequence or array of local frame configurations in global frame
    • [3xn]: [[x0, x1, ..., xn-1], [y0, y1, ..., yn-1], [φ0, φ1, ..., φn-1]]
    • [nx3]: [[x0, y0, φ0], [x1, y1, φ1], ..., [xn-1, yn-1, φn-1]]
    if the latter is provided, the ith xy pair will be mapped to the ith local frame
    :return: tr: an array of polar coordinate pairs, with shape identical to the input.
    • [2xn]: [[Θ0, Θ1, ..., Θn-1], [r0, r1, ..., rn-1]]
    • [nx2]: [[Θ0, r0], [Θ1, r1], ..., [Θn-1, rn-1]]
    """

    # Convert input to dimension [2xn]
    xy, v_flag, t_flag = _convert_to_mxn(xy, 2)

    # Convert input to dimension [3xn]
    qf, _, _ = _convert_to_mxn(qf, 3)

    # Translate data from global frame to local frame
    xy = xy - qf[0:2, :]

    # Convert to polar coordinates
    tr = cart2pol(xy)

    # Rotate
    tr[0, :] -= qf[2, :]

    # Match output shape to input shape
    tr = _match_shape_of_out_to_in(tr, v_flag, t_flag)

    return tr


# ----------------------------------------------------------------------------------------------------------------------
# Convert data in local polar coordinates to global cartesian coordinates
# ----------------------------------------------------------------------------------------------------------------------
def local_pol_2_global_cart(tr: np.ndarray |
                                tuple[float, float] |
                                tuple[tuple[float, float], ...] |
                                tuple[list[float, float], ...] |
                                list[float] |
                                list[tuple[float, float]] |
                                list[list[float, float]],
                            qf: np.ndarray |
                                tuple[float, float, float] |
                                list[float, float, float]) -> np.ndarray:
    """
    Convert polar data in a local frame, to cartesian data in a global frame.
    :param tr: a sequence or array of polar coordinate pairs
    • [2xn]: [[Θ0, Θ1, ..., Θn-1], [r0, r1, ..., rn-1]]
    • [nx2]: [[Θ0, r0], [Θ1, r1], ..., [Θn-1, rn-1]]
    :param qf: local frame configuration in the global frame
    • [3x1]: [x, y, φ]
    or a sequence or array of local frame configurations in global frame
    • [3xn]: [[x0, x1, ..., xn-1], [y0, y1, ..., yn-1], [φ0, φ1, ..., φn-1]]
    • [nx3]: [[x0, y0, φ0], [x1, y1, φ1], ..., [xn-1, yn-1, φn-1]]
    if the latter is provided, the ith tr pair will be mapped from the ith local frame
    :return: xy: an array of cartesian coordinate pairs, with shape identical to the input.
    • [2xn]: [[x0, x1, ..., xn-1], [y0, y1, ..., yn-1]]
    • [nx2]: [[x0, y0], [x1, y1], ..., [xn-1, yn-1]]

    """
    # Convert input to dimension [2xn]
    tr, v_flag, t_flag = _convert_to_mxn(tr, 2)

    # Convert input to dimension [3xn]
    qf, _, _ = _convert_to_mxn(qf, 3)

    # Convert data from polar coordinates to cartesian coordinates in local frame
    xy = pol2cart(tr)

    # Rotate
    xy = rot2d(xy, qf[2, :])

    # Translate
    xy = xy + qf[0:2, :]

    # Match output shape to input shape
    xy = _match_shape_of_out_to_in(xy, v_flag, t_flag)

    return xy

# ----------------------------------------------------------------------------------------------------------------------
# Circle Inversion (added locally 12/15/24)
# ----------------------------------------------------------------------------------------------------------------------
def circle_inv_cart(xy: np.ndarray |
                        tuple[float, float] |
                        tuple[tuple[float, float], ...] |
                        tuple[list[float], ...] |
                        list[float] |
                        list[tuple[float, float]] |
                        list[list[float]],
                    rad: float |
                         tuple[float, ...] |
                         list[float] |
                         np.ndarray) -> np.ndarray:
    """
    Invert input cartesian coordinate pairs about a circle of radius rad
    :param xy: a sequence or array of cartesian coordinate pairs, i.e.
    • [2xn]: [[x0, x1, ..., xn-1], [y0, y1, ..., yn-1]]
    • [nx2]: [[x0, y0], [x1, y1], ..., [xn-1, yn-1]]
    If the shape of xy is [nx2], and n!=2, xy is transposed before calculation.
    If xy is a tuple or list, it is converted to a Numpy array.
    :param rad: radius of the circle of inversion
    If the argument is a scalar, inversion is applied to all coordinate pairs.
    If the argument is vector-like, inversion of the ith angle is applied to the ith pair, and it must contain exactly n values.
    :return: an array of cartesian coordinate pairs [x, y], with shape identical to the input
    """

    # Convert input to dimension [2xn]
    xy, v_flag, t_flag = _convert_to_mxn(xy, 2)

    # Convert input to dimension [1xn]
    rad, _, _ = _convert_to_mxn(rad, 1)

    # Convert to polar coordinates
    tr = cart2pol(xy)

    # Invert about circle of radius rad
    tr[1, :] = rad ** 2 / tr[1, :]

    # Convert to cartesian coordinates
    xy = pol2cart(tr)

    # Match output shape to input shape
    xy = _match_shape_of_out_to_in(xy, v_flag, t_flag)

    return xy

# ----------------------------------------------------------------------------------------------------------------------
# Test Case
# ----------------------------------------------------------------------------------------------------------------------
def test_case():
    # Cartesian [x, y]
    xy = np.array(
        ((1.0, 0.707, 0.0),
         (0.0, 0.707, 1.0)),
        dtype=float
    )

    # polar [theta, rho]
    tr = np.array(
        ((0.0, 0.25 * np.pi, 0.5 * np.pi),
         (1.0, 1.0, 1.0)),
        dtype=float
    )

    # polar [theta, rho], where rho is in degrees
    tr_deg = np.array(
        ((0.0, 45.0, 90.0),
         (1.0, 1.0, 1.0)),
        dtype=float
    )

    # Cartesian frame configurations [x, y, phi]
    qf = np.array(
        ((0.0, 1.0, 1.0),
         (0.0, 1.0, 0.0),
         (np.pi, 0.0, 0.5 * np.pi)),
        dtype=float
    )

    # --------------------------------
    # rot2d(...)
    # --------------------------------
    print("# Test: rot2d(...)")
    val00 = rot2d(xy, tr[0, :])
    val01 = rot2d(xy.T, list(tr[0, :]))
    val02 = rot2d(xy, tr[0, 1])
    val03 = rot2d(xy[:, 1], tr[0, 1])
    val04 = rot2d(tuple(xy[:, 1]), tr[0, 1])
    val05 = rot2d(xy, tr_deg[0, :], deg=True)
    print(np.round(val05, 3))
    print("\n")

    # --------------------------------
    # cart2pol(...)
    # --------------------------------
    print("# Test: cart2pol(...)")
    val00 = cart2pol(xy)
    val01 = cart2pol(tuple(xy))
    val02 = cart2pol(list(xy.T))
    val03 = cart2pol(xy[:, 0])
    print(np.round(val03, 3))
    print("\n")

    # --------------------------------
    # pol2cart(...)
    # --------------------------------
    print("# Test: pol2cart(...)")
    val00 = pol2cart(tr)
    val01 = pol2cart(tuple(tr))
    val02 = pol2cart(list(tr.T))
    val03 = pol2cart(tr[:, 1])
    print(np.round(val03, 3))
    print("\n")

    # --------------------------------
    # pol2cart(...) ↔ cart2pol(...)
    # --------------------------------
    print("# Test: pol2cart(...) ↔ cart2pol(...)")
    val00 = pol2cart(cart2pol(xy))
    val01 = cart2pol(pol2cart(tr))
    val02 = pol2cart(cart2pol(xy[:, 0]))
    val03 = pol2cart(cart2pol(tuple(xy.T)))
    print(np.round(val03, 3))
    print("\n")

    # --------------------------------
    # global_cart_2_local_pol(...) ↔ local_pol_2_global_cart(...)
    # --------------------------------
    print("# Test: global_cart_2_local_pol(...) ↔ local_pol_2_global_cart(...)")
    val00 = local_pol_2_global_cart(tr=global_cart_2_local_pol(xy=xy, qf=qf), qf=qf)
    val01 = global_cart_2_local_pol(xy=local_pol_2_global_cart(tr=tr, qf=qf), qf=qf)
    print(np.round(val00, 3))
    print("\n")

    # --------------------------------
    # circle_inv_cart(...)
    # --------------------------------
    print("# Test: circle_inv_cart(...)")
    val00 = circle_inv_cart(xy=xy.T, rad=(2, 1, 2))
    print(np.round(val00, 3))
    print("\n")

# ----------------------------------------------------------------------------------------------------------------------
# Call to test case for validation purposes
# ---------------------------------------------------------------------------------------------------------------------
# Note: I think best practice is to save the test code in its own module, but I don't want to do that.
if __name__ == "__main__":
    test_case()