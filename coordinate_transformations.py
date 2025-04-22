"""
A Collection of python functions which perform coordinate transformations
• cart2pol
• pol2cart
• local_pol_2_global_cart
• global_cart_2_local_pol
• test case
"""
import numpy as np

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
    :param xy: a [2xn] or [2xn] array of cartesian coordinate pairs, i.e.
    • [2xn]: [[x0, x1, ..., xn-1], [y0, y1, ..., yn-1]]]
    • [nx2]: [[x0, y0], [x1, y1], ..., [xn-1, yn-1]]
    If the shape of xy is [nx2], and n!=2, xy is transposed before calculation.
    If xy is a tuple or list, it is converted to a Numpy array.
    :param th: angle of rotation about the origin.
    If the argument is a scalar, rotation is applied to all coordinate pairs.
    If the argument is vector-like, rotation of the ith angle is applied to the ith pair, and it must contain exactly n values.
    :param deg: Set to True if unit of measure of input angle is degrees; otherwise unit of measure is radians
    :return: an array of n rotated cartesian coordinate pairs, with shape identical to the input.
    """

    # Convert to numpy array
    if not isinstance(xy, np.ndarray):
        xy = np.array(xy)

    # if xy is vector, add dimension
    if max(xy.shape) == 1:
        xy = xy[:, None]

    # Ensure array is correct shape for calculation [2xn]
    transpose_flag = False
    if xy.shape[0] != 2 and xy.shape[1] == 2:
        transpose_flag = True
        xy = xy.T

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
    if transpose_flag:
        xy = xy.T

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
    :param xy: a [2xn] or [2xn] array of cartesian coordinate pairs, i.e.
    • [2xn]: [[x0, x1, ..., xn-1], [y0, y1, ..., yn-1]]]
    • [nx2]: [[x0, y0], [x1, y1], ..., [xn-1, yn-1]]
    If the shape of xy is [nx2], and n!=2, xy is transposed before calculation.
    If xy is a tuple or list, it is converted to a Numpy array.
    :return: an array of polar coordinate pairs [Θ, r], with shape identical to the input
    """

    # Convert to numpy array
    if not isinstance(xy, np.ndarray):
        xy = np.array(xy)

    # if xy is vector, add dimension
    vector_flag = False
    if len(xy.shape) == 1:
        vector_flag = True
        xy = xy[:, None]

    # Ensure array is correct shape for calculation [2xn]
    transpose_flag = False
    if xy.shape[0] != 2 and xy.shape[1] == 2:
        transpose_flag = True
        xy = xy.T

    # Transform
    tr = np.array(
        (np.arctan2(xy[1, :], xy[0, :]),
         np.hypot(xy[0, :], xy[1, :])),
        dtype=float
    )

    # Match output shape to input shape
    if transpose_flag:
        tr = tr.T

    if vector_flag:
        tr = tr.T.squeeze()

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
    :param tr: a sequence of polar coordinate pairs [[theta, rho], ...]
    :return: a numpy.ndarray of cartesian coordinate pairs [[x, y], ...]
    """
    if not isinstance(tr, np.ndarray):
        tr = np.array(tr)

    if len(tr.shape) == 1:
        tr = tr[None, :]

    x = tr[:, 1] * np.cos(tr[:, 0])
    y = tr[:, 1] * np.sin(tr[:, 0])

    return np.squeeze(np.stack((x, y), axis=1))


# ----------------------------------------------------------------------------------------------------------------------
# Convert data in global cartesian coordinates to local polar coordinates
# ----------------------------------------------------------------------------------------------------------------------
def global_cart_2_local_pol(xy: np.ndarray |
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
    Convert cartesian data in a global frame, to polar data in a local frame.
    :param xy: a sequence of cartesian coordinate pairs [[x, y], ...]
    :param qf: local frame configuration [x, y, Θ] in global frame
    :return: tr: a numpy.ndarray of polar coordinate pairs [[theta, rho], ...]
    """

    if not isinstance(xy, np.ndarray):
        xy = np.array(xy)

    if not isinstance(qf, np.ndarray):
        qf = np.array(qf)

    # Translate data from global frame to local frame
    xy = xy - qf[0:2]

    # Convert to polar coordinates
    tr = cart2pol(xy)

    # Add dimension if necessary; permits boradcast "rotate" operation
    if len(xy.shape) == 1:
        tr = tr[None, :]

    # Rotate
    tr = tr.T
    tr[0, :] -= qf[2]
    tr = tr.T

    return tr.squeeze()


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
    :param tr: a sequence of polar coordinate pairs [[theta, rho], ...]
    :param qf: local frame configuration [x, y, Θ] in global frame
    :return: a numpy.ndarray of cartesian coordinate pairs [[x, y], ...]
    """

    if not isinstance(tr, np.ndarray):
        tr = np.array(tr)

    if not isinstance(qf, np.ndarray):
        qf = np.array(qf)

    # Convert data from polar coordinates to cartesian coordinates in local frame
    xy = pol2cart(tr)

    # Rotate
    xy = rot2d(xy, float(qf[2]))

    # Translate
    xy = xy + qf[0:2]

    return xy.squeeze()

# ----------------------------------------------------------------------------------------------------------------------
# Circle Inversion (added locally 12/15/24)
# ----------------------------------------------------------------------------------------------------------------------
def circle_inv(p: float | tuple[float, ...] | list[float] | np.ndarray[float],
               r: float) -> np.ndarray:
    if not isinstance(p, np.ndarray):
        p = np.array(p)
    q = r ** 2 / p
    return q

# ----------------------------------------------------------------------------------------------------------------------
# Test Case
# ----------------------------------------------------------------------------------------------------------------------
def test_case():
    x0 = [1, 1, 0]
    x1 = [3, 4]
    x2 = [[3, 4], [5, 12], [8, 15], [7, 24], [20, 21]]
    x4 = [[1, np.sqrt(0.5), 0], [0, np.sqrt(0.5), 1]]

    t1 = 0.5 * np.pi
    t2 = [0.9273, 1.1760, 1.0808, 1.2870, 0.8097]
    t3 = [0, 0.25 * np.pi, 0.5 * np.pi]
    t4 = [0, 45, 90]

    r2 = [5, 13, 17, 25, 29]

    p1 = [0.9273, 5]
    p2 = np.stack((t2, r2), axis=1)

    x3 = np.add(x0[0:2], x2)

    print("--------------------------------")
    print("rot2d")
    print("--------------------------------")
    print("expected")
    print([-4, 3])
    print("actual")
    print(rot2d(xy=x1, th=t1, deg=False))

    print("expected")
    print([[0, float(-np.sqrt(0.5)), -1], [1, float(np.sqrt(0.5)), 0]])
    print("actual")
    print(np.round(rot2d(xy=x4, th=t1, deg=False), 2))

    print("expected")
    print([[1, 0, -1], [0, 1, 0]])
    print("actual")
    print(np.round(rot2d(xy=x4, th=t3, deg=False), 2))

    print("expected")
    print([[1, 0, -1], [0, 1, 0]])
    print("actual")
    print(np.round(rot2d(xy=x4, th=t4, deg=True), 2))

    print("--------------------------------")
    print("cart2pol() test")
    print("--------------------------------")
    print("expected")
    print(p2[0, :])
    print("actual")
    print(cart2pol([3, 4]))

    print("expected")
    print(p2)
    print("actual")
    print(cart2pol(x2))

    return

    print("--------------------------------")
    print("pol2cart() test")
    print("--------------------------------")
    print("expected")
    print(x1)
    print("actual")
    print(pol2cart(p1))
    print("expected")
    print(np.round(x2, 3))
    print("actual")
    print(np.round(pol2cart(p2), 3))

    print("--------------------------------")
    print("pol2cart() ↔ cart2pol() test")
    print("--------------------------------")
    print(np.round((x2, pol2cart(cart2pol(x2))), 3))
    print(np.round((p2, cart2pol(pol2cart(p2))), 3))

    print("--------------------------------")
    print("global_cart_2_local_pol() test")
    print("--------------------------------")
    print(np.round(p2[0], 3))
    print(np.round(global_cart_2_local_pol(xy=x3[0], qf=x0), 3))

    print(np.round(p2, 3))
    print(np.round(global_cart_2_local_pol(xy=x3, qf=x0), 3))

    print("--------------------------------")
    print("local_pol_2_global_cart() test")
    print("--------------------------------")
    print(np.round(x3[0], 3))
    print(np.round(local_pol_2_global_cart(tr=p2[0], qf=x0), 3))

    print(np.round(x3, 3))
    print(np.round(local_pol_2_global_cart(tr=p2, qf=x0), 3))


# ----------------------------------------------------------------------------------------------------------------------
# Call to test case for validation purposes
# ---------------------------------------------------------------------------------------------------------------------
# Note: I think best practice is to save the test code in its own module, but I don't want to do that.
# Note: Keep the call to test_case() commented out during actual use
if __name__ == "__main__":
    test_case()