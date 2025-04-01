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
              tuple[list[float, float], ...] |
              list[float] |
              list[tuple[float, float]] |
              list[list[float, float]],
          th: float):
    """
    Rotate 2d cartesian coordinates about the origin
    :param xy: a sequence of cartesian coordinate pairs [[x, y], ...]
    :param th: angle to about which to rotate xy1 (only one angle argument supported)
    :return: a numpy.ndarray of cartesian coordinate pairs [[x, y], ...]
    """
    if not isinstance(xy, np.ndarray):
        xy = np.array(xy)

    xy = np.matmul([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], xy.T)

    return xy.T


# ----------------------------------------------------------------------------------------------------------------------
# Cartesian to Polar
# ----------------------------------------------------------------------------------------------------------------------
def cart2pol(xy: np.ndarray |
                 tuple[float, float] |
                 tuple[tuple[float, float], ...] |
                 tuple[list[float, float], ...] |
                 list[float] |
                 list[tuple[float, float]] |
                 list[list[float, float]]) -> np.ndarray:
    """
    Convert from cartesian coordinates to polar coordinates in a common frame
    :param xy: a sequence of cartesian coordinate pairs [[x, y], ...]
    :return: a numpy.ndarray of polar coordinate pairs [[theta, rho], ...]
    """
    if not isinstance(xy, np.ndarray):
        xy = np.array(xy)

    if len(xy.shape) == 1:
        xy = xy[None, :]

    rho = np.hypot(xy[:, 0], xy[:, 1])
    theta = np.arctan2(xy[:, 1], xy[:, 0])

    return np.squeeze(np.stack((theta, rho), axis=1))


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
    t2 = [0.9273, 1.1760, 1.0808, 1.2870, 0.8097]
    r2 = [5, 13, 17, 25, 29]

    p1 = [0.9273, 5]
    p2 = np.stack((t2, r2), axis=1)

    x3 = np.add(x0[0:2], x2)

    print("--------------------------------")
    print("cart2pol() test")
    print("--------------------------------")
    print("expected")
    print(p2[0, :])
    print("actual")
    print(cart2pol(x1))
    print("expected")
    print(p2)
    print("actual")
    print(cart2pol(x2))

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