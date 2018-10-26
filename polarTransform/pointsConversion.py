import numpy as np


def getCartesianPoints(rTheta, center):
    """Convert list of polar points to cartesian points

    The returned points are not rounded to the nearest point. User must do that by hand if desired.

    Parameters
    ----------
    rTheta : (N, 2) or (2,) :class:`numpy.ndarray`
        List of cartesian points to convert to polar domain

        First column is r and second column is theta
    center : (2,) :class:`numpy.ndarray`
        Center to use for conversion to cartesian domain of polar points

        Format of center is (x, y)

    Returns
    -------
    cartesianPoints : (N, 2) :class:`numpy.ndarray`
        Corresponding cartesian points from cartesian :obj:`rTheta`

        First column is x and second column is y

    See Also
    --------
    :meth:`getCartesianPoints2`
    """
    if rTheta.ndim == 2:
        x = rTheta[:, 0] * np.cos(rTheta[:, 1]) + center[0]
        y = rTheta[:, 0] * np.sin(rTheta[:, 1]) + center[1]
    else:
        x = rTheta[0] * np.cos(rTheta[1]) + center[0]
        y = rTheta[0] * np.sin(rTheta[1]) + center[1]

    return np.array([x, y]).T


def getCartesianPoints2(r, theta, center):
    """Convert list of polar points to cartesian points

    The returned points are not rounded to the nearest point. User must do that by hand if desired.

    Parameters
    ----------
    r : (N,) :class:`numpy.ndarray`
        List of polar r points to convert to cartesian domain
    theta : (N,) :class:`numpy.ndarray`
        List of polar theta points to convert to cartesian domain
    center : (2,) :class:`numpy.ndarray`
        Center to use for conversion to cartesian domain of polar points

        Format of center is (x, y)

    Returns
    -------
    x : (N,) :class:`numpy.ndarray`
        Corresponding x points from polar :obj:`r` and :obj:`theta`
    y : (N,) :class:`numpy.ndarray`
        Corresponding y points from polar :obj:`r` and :obj:`theta`

    See Also
    --------
    :meth:`getCartesianPoints`
    """
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]

    return x, y


def getPolarPoints(xy, center):
    """Convert list of cartesian points to polar points

    The returned points are not rounded to the nearest point. User must do that by hand if desired.

    Parameters
    ----------
    xy : (N, 2) or (2,) :class:`numpy.ndarray`
        List of cartesian points to convert to polar domain

        First column is x and second column is y
    center : (2,) :class:`numpy.ndarray`
        Center to use for conversion to polar domain of cartesian points

        Format of center is (x, y)

    Returns
    -------
    polarPoints : (N, 2) :class:`numpy.ndarray`
        Corresponding polar points from cartesian :obj:`xy`

        First column is r and second column is theta

    See Also
    --------
    :meth:`getPolarPoints2`
    """
    if xy.ndim == 2:
        cX, cY = xy[:, 0] - center[0], xy[:, 1] - center[1]
    else:
        cX, cY = xy[0] - center[0], xy[1] - center[1]

    r = np.sqrt(cX ** 2 + cY ** 2)
    theta = np.arctan2(cY, cX)

    # Make range of theta 0 -> 2pi instead of -pi -> pi
    # According to StackOverflow, this is the fastest method:
    # https://stackoverflow.com/questions/37358016/numpy-converting-range-of-angles-from-pi-pi-to-0-2pi
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)

    return np.array([r, theta]).T


def getPolarPoints2(x, y, center):
    """Convert list of cartesian points to polar points

    The returned points are not rounded to the nearest point. User must do that by hand if desired.

    Parameters
    ----------
    x : (N,) :class:`numpy.ndarray`
        List of cartesian x points to convert to polar domain
    y : (N,) :class:`numpy.ndarray`
        List of cartesian y points to convert to polar domain
    center : (2,) :class:`numpy.ndarray`
        Center to use for conversion to polar domain of cartesian points

        Format of center is (x, y)

    Returns
    -------
    r : (N,) :class:`numpy.ndarray`
        Corresponding radii points from cartesian :obj:`x` and :obj:`y`
    theta : (N,) :class:`numpy.ndarray`
        Corresponding theta points from cartesian :obj:`x` and :obj:`y`

    See Also
    --------
    :meth:`getPolarPoints`
    """
    cX, cY = x - center[0], y - center[1]

    r = np.sqrt(cX ** 2 + cY ** 2)

    theta = np.arctan2(cY, cX)

    # Make range of theta 0 -> 2pi instead of -pi -> pi
    # According to StackOverflow, this is the fastest method:
    # https://stackoverflow.com/questions/37358016/numpy-converting-range-of-angles-from-pi-pi-to-0-2pi
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)

    return r, theta


def getPolarPointsImage(points, settings):
    """Convert list of cartesian points from image to polar image points based on transform metadata

    .. warning::
        Cleaner and more succinct to use :meth:`ImageTransform.getPolarPointsImage`

    .. note::
        This does **not** convert from cartesian to polar points, but rather converts pixels from cartesian image to
        pixels from polar image using :class:`ImageTransform`.

    The returned points are not rounded to the nearest point. User must do that by hand if desired.

    Parameters
    ----------
    points : (N, 2) or (2,) :class:`numpy.ndarray`
        List of cartesian points to convert to polar domain

        First column is x and second column is y
    settings : :class:`ImageTransform`
        Contains metadata for conversion from polar to cartesian domain

        Settings contains many of the arguments in :func:`convertToPolarImage` and :func:`convertToCartesianImage` and
        provides an easy way of passing these parameters along without having to specify them all again.

    Returns
    -------
    polarPoints : (N, 2) or (2,) :class:`numpy.ndarray`
        Corresponding polar points from cartesian :obj:`points` using :obj:`settings`

    See Also
    --------
    :meth:`ImageTransform.getPolarPointsImage`, :meth:`getPolarPoints`, :meth:`getPolarPoints2`
    """
    # Convert points to NumPy array
    points = np.asanyarray(points)

    # If there is only one point specified and number of dimensions is only one, then make the array a 1x2 array so that
    # points[:, 0/1] will not throw an error
    if points.ndim == 1 and points.shape[0] == 2:
        points = np.expand_dims(points, axis=0)
        needSqueeze = True
    else:
        needSqueeze = False

    # This is used to scale the result of the radius to get the appropriate Cartesian value
    scaleRadius = settings.polarImageSize[0] / (settings.finalRadius - settings.initialRadius)

    # This is used to scale the result of the angle to get the appropriate Cartesian value
    scaleAngle = settings.polarImageSize[1] / (settings.finalAngle - settings.initialAngle)

    # Take cartesian grid and convert to polar coordinates
    polarPoints = getPolarPoints(points, settings.center)

    # Offset the radius by the initial source radius
    polarPoints[:, 0] = polarPoints[:, 0] - settings.initialRadius

    # Offset the theta angle by the initial source angle
    # The theta values may go past 2pi, so they are looped back around by taking modulo with 2pi.
    # Note: This assumes initial source angle is positive
    # theta = np.mod(theta - initialAngle + 2 * np.pi, 2 * np.pi)
    polarPoints[:, 1] = np.mod(polarPoints[:, 1] - settings.initialAngle + 2 * np.pi, 2 * np.pi)

    # Scale the radius using scale factor
    # Scale the angle from radians to pixels using scale factor
    polarPoints = polarPoints * [scaleRadius, scaleAngle]

    if needSqueeze:
        return np.squeeze(polarPoints)
    else:
        return polarPoints


def getCartesianPointsImage(points, settings):
    """Convert list of polar points from image to cartesian image points based on transform metadata

    .. warning::
        Cleaner and more succinct to use :meth:`ImageTransform.getCartesianPointsImage`

    .. note::
        This does **not** convert from polar to cartesian points, but rather converts pixels from polar image to
        pixels from cartesian image using :class:`ImageTransform`.

    The returned points are not rounded to the nearest point. User must do that by hand if desired.

    Parameters
    ----------
    points : (N, 2) or (2,) :class:`numpy.ndarray`
        List of polar points to convert to cartesian domain

        First column is r and second column is theta
    settings : :class:`ImageTransform`
        Contains metadata for conversion from polar to cartesian domain

        Settings contains many of the arguments in :func:`convertToPolarImage` and :func:`convertToCartesianImage` and
        provides an easy way of passing these parameters along without having to specify them all again.

    Returns
    -------
    cartesianPoints : (N, 2) or (2,) :class:`numpy.ndarray`
        Corresponding cartesian points from polar :obj:`points` using :obj:`settings`

    See Also
    --------
    :meth:`ImageTransform.getCartesianPointsImage`, :meth:`getCartesianPoints`, :meth:`getCartesianPoints2`
    """
    # Convert points to NumPy array
    points = np.asanyarray(points)

    # If there is only one point specified and number of dimensions is only one, then make the array a 1x2 array so that
    # points[:, 0/1] will not throw an error
    if points.ndim == 1 and points.shape[0] == 2:
        points = np.expand_dims(points, axis=0)
        needSqueeze = True
    else:
        needSqueeze = False

    # This is used to scale the result of the radius to get the appropriate Cartesian value
    scaleRadius = settings.polarImageSize[0] / (settings.finalRadius - settings.initialRadius)

    # This is used to scale the result of the angle to get the appropriate Cartesian value
    scaleAngle = settings.polarImageSize[1] / (settings.finalAngle - settings.initialAngle)

    # Create a new copy of the points variable because we are going to change it and don't want the points parameter to
    # change outside of this function
    points = points.copy()

    # Scale the radius using scale factor
    # Scale the angle from radians to pixels using scale factor
    points = points / [scaleRadius, scaleAngle]

    # Offset the radius by the initial source radius
    points[:, 0] = points[:, 0] + settings.initialRadius

    # Offset the theta angle by the initial source angle
    # The theta values may go past 2pi, so they are looped back around by taking modulo with 2pi.
    # Note: This assumes initial source angle is positive
    # theta = np.mod(theta - initialAngle + 2 * np.pi, 2 * np.pi)
    points[:, 1] = np.mod(points[:, 1] + settings.initialAngle + 2 * np.pi, 2 * np.pi)

    # Take cartesian grid and convert to polar coordinates
    cartesianPoints = getCartesianPoints(points, settings.center)

    if needSqueeze:
        return np.squeeze(cartesianPoints)
    else:
        return cartesianPoints
