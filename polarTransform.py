import numpy as np
import scipy.interpolate
import scipy.ndimage
import skimage.util


class ImageTransform:
    def __init__(self, center, initialRadius, finalRadius, initialAngle, finalAngle, cartesianImageSize,
                 polarImageSize):
        """Polar and Cartesian Transform Metadata

        ImageTransform contains polar and cartesian transform metadata for the conversion between the two domains.
        This metadata is stored in a class to allow for easy conversion between the domains.

        Parameters
        ----------
        center : (2,) :class:`numpy.ndarray` of :class:`int`
            Specifies the center in the cartesian image to use as the origin in polar domain. The center in the
            cartesian domain will be (0, 0) in the polar domain.

            The center is structured as (x, y) where the first item is the x-coordinate and second item is the
            y-coordinate.
        initialRadius : :class:`int`
            Starting radius in pixels from the center of the cartesian image in the polar image

            The polar image begins at this radius, i.e. the first row of the polar image corresponds to this
            starting radius.
        finalRadius : :class:`int`, optional
            Final radius in pixels from the center of the cartesian image in the polar image

            The polar image ends at this radius, i.e. the last row of the polar image corresponds to this ending
            radius.
        initialAngle : :class:`float`, optional
            Starting angle in radians in the polar image

            The polar image begins at this angle, i.e. the first column of the polar image corresponds to this
            starting angle.

            Radian angle is with respect to the x-axis and rotates counter-clockwise. The angle should be in the range
            of 0 to :math:`2\pi`.
        finalAngle : :class:`float`, optional
            Final angle in radians in the polar image

            The polar image ends at this angle, i.e. the last column of the polar image corresponds to this
            ending angle.

            Radian angle is with respect to the x-axis and rotates counter-clockwise. The angle should be in the range
            of 0 to :math:`2\pi`.
        cartesianImageSize : (2,) :class:`tuple` of :class:`int`
            Size of cartesian image
        polarImageSize : (2,) :class:`tuple` of :class:`int`
            Size of polar image
        """
        self.center = center
        self.initialRadius = initialRadius
        self.finalRadius = finalRadius
        self.initialAngle = initialAngle
        self.finalAngle = finalAngle
        self.cartesianImageSize = cartesianImageSize
        self.polarImageSize = polarImageSize

    def convertToPolarImage(self, image, order=3, border='constant', borderVal=0.0):
        """Convert cartesian image to polar image.

        Using a cartesian image, this function creates a polar domain image where the first dimension is radius and
        second dimension is the angle. This function is versatile because it allows different starting and stopping
        radii and angles to extract the polar region you are interested in.

        .. note::
            Traditionally images are loaded such that the origin is in the upper-left hand corner. In these cases the
            :obj:`initialAngle` and :obj:`finalAngle` will rotate clockwise from the x-axis. For simplicitly, it is
            recommended to flip the image along first dimension before passing to this function.

        Parameters
        ----------
        image : (N, M, 3) or (N, M, 4) :class:`numpy.ndarray`
            Cartesian image to convert to polar domain

            .. note::
                If an alpha band (4th channel of image is present, then it will be ignored during polar conversion. The
                resulting polar image will contain four channels but the alpha channel will be all fully on.
        order : :class:`int` (0-5), optional
            The order of the spline interpolation, default is 3. The order has to be in the range 0-5.

            The following orders have special names:

                * 0 - nearest neighbor
                * 1 - bilinear
                * 3 - bicubic
        border : {'constant', 'nearest', 'wrap', 'reflect'}, optional
            Polar points outside the cartesian image boundaries are filled according to the given mode.

            Default is 'constant'

            The following table describes the mode and expected output when seeking past the boundaries. The input
            column is the 1D input array whilst the extended columns on either side of the input array correspond to
            the expected values for the given mode if one extends past the boundaries.

            .. table:: Valid border modes and expected output
                :widths: auto

                ==========  ======  =================  ======
                Mode        Ext.    Input              Ext.
                ==========  ======  =================  ======
                mirror      4 3 2   1 2 3 4 5 6 7 8    7 6 5
                reflect     3 2 1   1 2 3 4 5 6 7 8    8 7 6
                nearest     1 1 1   1 2 3 4 5 6 7 8    8 8 8
                constant    0 0 0   1 2 3 4 5 6 7 8    0 0 0
                wrap        6 7 8   1 2 3 4 5 6 7 8    1 2 3
                ==========  ======  =================  ======

            Refer to :func:`scipy.ndimage.map_coordinates` for more details on this argument.
        borderVal : same datatype as :obj:`image`, optional
            Value used for polar points outside the cartesian image boundaries if :obj:`border` = 'constant'.

            Default is 0.0

        Returns
        -------
        polarImage : (N, M, 3) or (N, M, 4) :class:`numpy.ndarray`
            Polar image where first dimension is radii and second dimension is angle
        """
        image, ptSettings = convertToPolarImage(image, order=order, border=border, borderVal=borderVal, settings=self)
        return image

    def convertToCartesianImage(self, image, order=3, border='constant', borderVal=0.0):
        """Convert polar image to cartesian image.

        Using a polar image, this function creates a cartesian image. This function is versatile because it can
        automatically calculate an appropiate cartesian image size and center given the polar image. In addition,
        parameters for converting to the polar domain are necessary for the conversion back to the cartesian domain.

        Parameters
        ----------
        image : (N, M, 3) or (N, M, 4) :class:`numpy.ndarray`
            Polar image to convert to cartesian domain

            .. note::
                If an alpha band (4th channel of image is present, then it will be ignored during cartesian conversion.
                The resulting polar image will contain four channels but the alpha channel will be all fully on.
        order : :class:`int` (0-5), optional
            The order of the spline interpolation, default is 3. The order has to be in the range 0-5.

            The following orders have special names:

                * 0 - nearest neighbor
                * 1 - bilinear
                * 3 - bicubic
        border : {'constant', 'nearest', 'wrap', 'reflect'}, optional
            Polar points outside the cartesian image boundaries are filled according to the given mode.

            Default is 'constant'

            The following table describes the mode and expected output when seeking past the boundaries. The input
            column is the 1D input array whilst the extended columns on either side of the input array correspond to
            the expected values for the given mode if one extends past the boundaries.

            .. table:: Valid border modes and expected output
                :widths: auto

                ==========  ======  =================  ======
                Mode        Ext.    Input              Ext.
                ==========  ======  =================  ======
                mirror      4 3 2   1 2 3 4 5 6 7 8    7 6 5
                reflect     3 2 1   1 2 3 4 5 6 7 8    8 7 6
                nearest     1 1 1   1 2 3 4 5 6 7 8    8 8 8
                constant    0 0 0   1 2 3 4 5 6 7 8    0 0 0
                wrap        6 7 8   1 2 3 4 5 6 7 8    1 2 3
                ==========  ======  =================  ======

            Refer to :func:`scipy.ndimage.map_coordinates` for more details on this argument.
        borderVal : same datatype as :obj:`image`, optional
            Value used for polar points outside the cartesian image boundaries if :obj:`border` = 'constant'.

            Default is 0.0

        Returns
        -------
        cartesianImage : (N, M, 3) or (N, M, 4) :class:`numpy.ndarray`
            Cartesian image

        See Also
        --------
        :meth:`convertToCartesianImage`
        """
        image, ptSettings = convertToCartesianImage(image, order=order, border=border, borderVal=borderVal,
                                                    settings=self)
        return image

    def getPolarPointsImage(self, points):
        """Convert list of cartesian points from image to polar image points based on transform metadata

        .. note::
            This does **not** convert from cartesian to polar points, but rather converts pixels from cartesian image to
            pixels from polar image using :class:`ImageTransform`.

        The returned points are not rounded to the nearest point. User must do that by hand if desired.

        Parameters
        ----------
        points : (N, 2) or (2,) :class:`numpy.ndarray`
            List of cartesian points to convert to polar domain

            First column is x and second column is y

        Returns
        -------
        polarPoints : (N, 2) or (2,) :class:`numpy.ndarray`
            Corresponding polar points from cartesian :obj:`points` using :class:`ImageTransform`

        See Also
        --------
        :meth:`getPolarPointsImage`, :meth:`getPolarPoints`, :meth:`getPolarPoints2`
        """

        return getPolarPointsImage(points, self)

    def getCartesianPointsImage(self, points):
        """Convert list of polar points from image to cartesian image points based on transform metadata

        .. note::
            This does **not** convert from polar to cartesian points, but rather converts pixels from polar image to
            pixels from cartesian image using :class:`ImageTransform`.

        The returned points are not rounded to the nearest point. User must do that by hand if desired.

        Parameters
        ----------
        points : (N, 2) or (2,) :class:`numpy.ndarray`
            List of polar points to convert to cartesian domain

            First column is r and second column is theta

        Returns
        -------
        cartesianPoints : (N, 2) or (2,) :class:`numpy.ndarray`
            Corresponding cartesian points from polar :obj:`points` using :class:`ImageTransform`

        See Also
        --------
        :meth:`getCartesianPointsImage`, :meth:`getCartesianPoints`, :meth:`getCartesianPoints2`
        """
        return getCartesianPointsImage(points, self)

    def __repr__(self):
        return 'ImageTransform(center=%s, initialRadius=%i, finalRadius=%i, initialAngle=%f, finalAngle=%f, ' \
               'cartesianImageSize=%s, polarImageSize=%s)' % (
                   self.center, self.initialRadius, self.finalRadius, self.initialAngle, self.finalAngle,
                   self.cartesianImageSize, self.polarImageSize)

    def __str__(self):
        return self.__repr__()


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


def convertToPolarImage(image, center=None, initialRadius=None, finalRadius=None, initialAngle=None, finalAngle=None,
                        radiusSize=None, angleSize=None, order=3, border='constant', borderVal=0.0,
                        settings=None):
    """Convert cartesian image to polar image.

    Using a cartesian image, this function creates a polar domain image where the first dimension is radius and
    second dimension is the angle. This function is versatile because it allows different starting and stopping
    radii and angles to extract the polar region you are interested in.

    .. note::
        Traditionally images are loaded such that the origin is in the upper-left hand corner. In these cases the
        :obj:`initialAngle` and :obj:`finalAngle` will rotate clockwise from the x-axis. For simplicitly, it is
        recommended to flip the image along first dimension before passing to this function.

    Parameters
    ----------
    image : (N, M, 3) or (N, M, 4) :class:`numpy.ndarray`
        Cartesian image to convert to polar domain

        .. note::
            If an alpha band (4th channel of image is present, then it will be ignored during polar conversion. The
            resulting polar image will contain four channels but the alpha channel will be all fully on.
    center : (2,) :class:`list`, :class:`tuple` or :class:`numpy.ndarray` of :class:`int`, optional
        Specifies the center in the cartesian image to use as the origin in polar domain. The center in the
        cartesian domain will be (0, 0) in the polar domain.

        The center is structured as (x, y) where the first item is the x-coordinate and second item is the y-coordinate.

        If center is not set, then it will default to ``round(image.shape[::-1] / 2)``.
    initialRadius : :class:`int`, optional
        Starting radius in pixels from the center of the cartesian image that will appear in the polar image

        The polar image will begin at this radius, i.e. the first row of the polar image will correspond to this
        starting radius.

        If initialRadius is not set, then it will default to ``0``.
    finalRadius : :class:`int`, optional
        Final radius in pixels from the center of the cartesian image that will appear in the polar image

        The polar image will end at this radius, i.e. the last row of the polar image will correspond to this ending
        radius.

        .. note::
            The polar image will **not** include this radius. It will include all radii starting
            from initial to final radii **excluding** the final radius. Rather, it will stop one step size before
            the final radius. Assuming the radial resolution (see :obj:`radiusSize`) is small enough, this should not
            matter.

        If finalRadius is not set, then it will default to the maximum radius of the cartesian image. Using the
        furthest corner from the center, the finalRadius can be calculated as:

        .. math::
            finalRadius = \sqrt{((X_{max} - X_{center})^2 + (Y_{max} - Y_{center})^2)}
    initialAngle : :class:`float`, optional
        Starting angle in radians that will appear in the polar image

        The polar image will begin at this angle, i.e. the first column of the polar image will correspond to this
        starting angle.

        Radian angle is with respect to the x-axis and rotates counter-clockwise. The angle should be in the range of
        0 to :math:`2\pi`.

        If initialAngle is not set, then it will default to ``0.0``.
    finalAngle : :class:`float`, optional
        Final angle in radians that will appear in the polar image

        The polar image will end at this angle, i.e. the last column of the polar image will correspond to this
        ending angle.

        .. note::
            The polar image will **not** include this angle. It will include all angle starting
            from initial to final angle **excluding** the final angle. Rather, it will stop one step size before
            the final angle. Assuming the angular resolution (see :obj:`angleSize`) is small enough, this should not
            matter.

        Radian angle is with respect to the x-axis and rotates counter-clockwise. The angle should be in the range of
        0 to :math:`2\pi`.

        If finalAngle is not set, then it will default to :math:`2\pi`.
    radiusSize : :class:`int`, optional
        Size of polar image for radial (1st) dimension

        This in effect determines the resolution of the radial dimension of the polar image based on the
        :obj:`initialRadius` and :obj:`finalRadius`. Resolution can be calculated using equation below in radial
        px per cartesian px:

        .. math::
            radialResolution = \\frac{radiusSize}{finalRadius - initialRadius}

        If radiusSize is not set, then it will default to the minimum size necessary to ensure that image information
        is not lost in the transformation. The minimum resolution necessary can be found by finding the smallest
        change in radius from two connected pixels in the cartesian image. Through experimentation, there is a
        surprisingly close relationship between the maximum difference from width or height of the cartesian image to
        the :obj:`center` times two.

        The radiusSize is calculated based on this relationship and is proportional to the :obj:`initialRadius` and
        :obj:`finalRadius` given.
    angleSize : :class:`int`, optional
        Size of polar image for angular (2nd) dimension

        This in effect determines the resolution of the angular dimension of the polar image based on the
        :obj:`initialAngle` and :obj:`finalAngle`. Resolution can be calculated using equation below in angular
        px per cartesian px:

        .. math::
            angularResolution = \\frac{angleSize}{finalAngle - initialAngle}

        If angleSize is not set, then it will default to the minimum size necessary to ensure that image information
        is not lost in the transformation. The minimum resolution necessary can be found by finding the smallest
        change in angle from two connected pixels in the cartesian image.

        For a cartesian image with either dimension greater than 500px, the angleSize is set to be **two** times larger
        than the largest dimension proportional to :obj:`initialAngle` and :obj:`finalAngle`. Otherwise, for a
        cartesian image with both dimensions less than 500px, the angleSize is set to be **four** times larger the
        largest dimension proportional to :obj:`initialAngle` and :obj:`finalAngle`.

        .. note::
            The above logic **estimates** the necessary angleSize to reduce image information loss. No algorithm
            currently exists for determining the required angleSize.
    order : :class:`int` (0-5), optional
        The order of the spline interpolation, default is 3. The order has to be in the range 0-5.

        The following orders have special names:

            * 0 - nearest neighbor
            * 1 - bilinear
            * 3 - bicubic
    border : {'constant', 'nearest', 'wrap', 'reflect'}, optional
        Polar points outside the cartesian image boundaries are filled according to the given mode.

        Default is 'constant'

        The following table describes the mode and expected output when seeking past the boundaries. The input column
        is the 1D input array whilst the extended columns on either side of the input array correspond to the expected
        values for the given mode if one extends past the boundaries.

        .. table:: Valid border modes and expected output
            :widths: auto

            ==========  ======  =================  ======
            Mode        Ext.    Input              Ext.
            ==========  ======  =================  ======
            mirror      4 3 2   1 2 3 4 5 6 7 8    7 6 5
            reflect     3 2 1   1 2 3 4 5 6 7 8    8 7 6
            nearest     1 1 1   1 2 3 4 5 6 7 8    8 8 8
            constant    0 0 0   1 2 3 4 5 6 7 8    0 0 0
            wrap        6 7 8   1 2 3 4 5 6 7 8    1 2 3
            ==========  ======  =================  ======

        Refer to :func:`scipy.ndimage.map_coordinates` for more details on this argument.
    borderVal : same datatype as :obj:`image`, optional
        Value used for polar points outside the cartesian image boundaries if :obj:`border` = 'constant'.

        Default is 0.0
    settings : :class:`ImageTransform`, optional
        Contains metadata for conversion between polar and cartesian image.

        Settings contains many of the arguments in :func:`convertToPolarImage` and :func:`convertToCartesianImage` and
        provides an easy way of passing these parameters along without having to specify them all again.

        .. warning::
            Cleaner and more succint to use :meth:`ImageTransform.convertToPolarImage`

        If settings is not specified, then the other arguments are used in this function and the defaults will be
        calculated if necessary. If settings is given, then the values from settings will be used.

    Returns
    -------
    polarImage : (N, M, 3) or (N, M, 4) :class:`numpy.ndarray`
        Polar image where first dimension is radii and second dimension is angle
    settings : :class:`ImageTransform`
        Contains metadata for conversion between polar and cartesian image.

        Settings contains many of the arguments in :func:`convertToPolarImage` and :func:`convertToCartesianImage` and
        provides an easy way of passing these parameters along without having to specify them all again.
    """

    # Determines whether there are multiple bands or channels in image by checking for 3rd dimension
    isMultiChannel = image.ndim == 3

    # Create settings if none are given
    if settings is None:
        # If center is not specified, set to the center of the image
        # Image shape is reversed because center is specified as x,y and shape is r,c.
        # Otherwise, make sure the center is a Numpy array
        if center is None:
            center = (np.array(image.shape[1::-1]) / 2).astype(int)
        else:
            center = np.array(center)

        # Initial radius is zero if none is selected
        if initialRadius is None:
            initialRadius = 0

        # Calculate the maximum radius possible
        # Get four corners (indices) of the cartesian image
        # Convert the corners to polar and get the largest radius
        # This will be the maximum radius to represent the entire image in polar
        corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * image.shape[0:2]
        radii, _ = getPolarPoints2(corners[:, 1], corners[:, 0], center)
        maxRadius = np.ceil(radii.max()).astype(int)

        if finalRadius is None:
            finalRadius = maxRadius

        # Initial angle of zero if none is selected
        if initialAngle is None:
            initialAngle = 0

        # Final radius is the size of the image so that all points from cartesian are on the polar image
        # Final angle is 2pi to loop throughout entire image
        if finalAngle is None:
            finalAngle = 2 * np.pi

        # If no radius size is given, then the size will be set to make the radius size twice the size of the largest
        # dimension of the image
        # There is a surprisingly close relationship between the maximum difference from
        # width/height of image to center times two.
        # The radius size is proportional to the final radius and initial radius
        if radiusSize is None:
            cross = np.array([[image.shape[1] - 1, center[1]], [0, center[1]], [center[0], image.shape[0] - 1],
                              [center[0], 0]])

            radiusSize = np.ceil(np.abs(cross - center).max() * 2 * (finalRadius - initialRadius) / maxRadius) \
                .astype(int)

        # Make the angle size be twice the size of largest dimension for images above 500px, otherwise
        # use a factor of 4x.
        # This angle size is proportional to the initial and final angle.
        # This was experimentally determined to yield the best resolution
        # The actual answer for the necessary angle size to represent all of the pixels is
        # (finalAngle - initialAngle) / (min(arctan(y / x) - arctan((y - 1) / x)))
        # Where the coordinates used in min are the four corners of the cartesian image with the center
        # subtracted from it. The minimum will be the corner that is the furthest away from the center
        # TODO Find a better solution to determining default angle size (optimum?)
        if angleSize is None:
            maxSize = np.max(image.shape)

            if maxSize > 500:
                angleSize = int(2 * np.max(image.shape) * (finalAngle - initialAngle) / (2 * np.pi))
            else:
                angleSize = int(4 * np.max(image.shape) * (finalAngle - initialAngle) / (2 * np.pi))

        # Create the settings
        settings = ImageTransform(center, initialRadius, finalRadius, initialAngle, finalAngle, image.shape[0:2],
                                  (radiusSize, angleSize))

    # Create radii from start to finish with radiusSize, do same for theta
    # Then create a 2D grid of radius and theta using meshgrid
    # Set endpoint to False to NOT include the final sample specified. Think of it like this, if you ask to count from
    # 0 to 30, that is 31 numbers not 30. Thus, we count 0...29 to get 30 numbers.
    radii = np.linspace(settings.initialRadius, settings.finalRadius, settings.polarImageSize[0], endpoint=False)
    theta = np.linspace(settings.initialAngle, settings.finalAngle, settings.polarImageSize[1], endpoint=False)
    r, theta = np.meshgrid(radii, theta)

    # Take polar  grid and convert to cartesian coordinates
    xCartesian, yCartesian = getCartesianPoints2(r, theta, settings.center)

    # Flatten the desired x/y cartesian points into one 2xN array
    desiredCoords = np.vstack((yCartesian.flatten(), xCartesian.flatten()))

    # If border is set to constant, then pad the image by the edges by 3 pixels.
    # If one tries to convert back to cartesian without the borders padded then the border of the cartesian image will
    # be corrupted because it will average the pixels with the border value
    if border == 'constant':
        # Pad image by 3 pixels and then offset all of the desired coordinates by 3
        image = np.pad(image, ((3, 3), (3, 3), (0, 0)) if isMultiChannel else 3, 'edge')
        desiredCoords += 3

    # Retrieve polar image using map_coordinates. Returns a linear array of the values that
    # must be reshaped into the desired size
    # For multiple channels, repeat this process for each band and concatenate them at end
    # Take the transpose of the polar image such that first dimension is radius and second
    # dimension is theta.
    if isMultiChannel:
        polarImages = []

        # Assume that there are at least 3 bands in 3D matrix
        for k in range(3):
            polarImage = scipy.ndimage.map_coordinates(image[:, :, k], desiredCoords, mode=border, cval=borderVal,
                                                       order=order).reshape(r.shape).T
            polarImages.append(polarImage)

        # If there are 4 bands, then assume the 4th band is alpha
        # We do not want to interpolate the transparency so we just make it all fully opaque
        if image.shape[2] == 4:
            imin, imax = skimage.util.dtype_limits(polarImages[0], False)
            polarImage = np.full_like(polarImages[0], imax)
            polarImages.append(polarImage)

        polarImage = np.dstack(polarImages)
    else:
        polarImage = scipy.ndimage.map_coordinates(image, desiredCoords, mode=border, cval=borderVal,
                                                   order=order).reshape(r.shape).T

    return polarImage, settings


def convertToCartesianImage(image, center=None, initialRadius=None,
                            finalRadius=None, initialAngle=None,
                            finalAngle=None, imageSize=None, order=3, border='constant',
                            borderVal=0.0, settings=None):
    """Convert polar image to cartesian image.

    Using a polar image, this function creates a cartesian image. This function is versatile because it can
    automatically calculate an appropiate cartesian image size and center given the polar image. In addition,
    parameters for converting to the polar domain are necessary for the conversion back to the cartesian domain.

    Parameters
    ----------
    image : (N, M, 3) or (N, M, 4) :class:`numpy.ndarray`
        Polar image to convert to cartesian domain

        .. note::
            If an alpha band (4th channel of image is present, then it will be ignored during cartesian conversion. The
            resulting polar image will contain four channels but the alpha channel will be all fully on.
    center : :class:`str` or (2,) :class:`list`, :class:`tuple` or :class:`numpy.ndarray` of :class:`int`, optional
        Specifies the center in the cartesian image to use as the origin in polar domain. The center in the
        cartesian domain will be (0, 0) in the polar domain.

        If center is not set, then it will default to ``middle-middle``. If the image size is :obj:`None`, the
        center is calculated after the image size is determined.

        For relative positioning within the image, center can be one of the string values in the table below. The
        quadrant column contains the visible quadrants for the given center. initialAngle and finalAngle must contain
        at least one of the quadrants, otherwise an error will be thrown because the resulting cartesian image is blank.
        An example cartesian image is given below with annotations to what the center will be given a center string.

        .. table:: Valid center strings
            :widths: auto

            ================  ===============  ====================
                 Value            Quadrant       Location in image
            ================  ===============  ====================
            top-left          IV               1
            top-middle        III, IV          2
            top-right         III              3
            middle-left       I, IV            4
            middle-middle     I, II, III, IV   5
            middle-right      II, III          6
            bottom-left       I                7
            bottom-middle     I, II            8
            bottom-right      II               9
            ================  ===============  ====================

        .. image:: _static/centerAnnotations.png
            :alt: Center locations for center strings
    initialRadius : :class:`int`, optional
        Starting radius in pixels from the center of the cartesian image in the polar image

        The polar image begins at this radius, i.e. the first row of the polar image corresponds to this
        starting radius.

        If initialRadius is not set, then it will default to ``0``.
    finalRadius : :class:`int`, optional
        Final radius in pixels from the center of the cartesian image in the polar image

        The polar image ends at this radius, i.e. the last row of the polar image corresponds to this ending
        radius.

        .. note::
            The polar image does **not** include this radius. It includes all radii starting
            from initial to final radii **excluding** the final radius. Rather, it will stop one step size before
            the final radius. Assuming the radial resolution (see :obj:`radiusSize`) is small enough, this should not
            matter.

        If finalRadius is not set, then it will default to the maximum radius which is the size of the radial (1st)
        dimension of the polar image.
    initialAngle : :class:`float`, optional
        Starting angle in radians in the polar image

        The polar image begins at this angle, i.e. the first column of the polar image corresponds to this
        starting angle.

        Radian angle is with respect to the x-axis and rotates counter-clockwise. The angle should be in the range of
        0 to :math:`2\pi`.

        If initialAngle is not set, then it will default to ``0.0``.
    finalAngle : :class:`float`, optional
        Final angle in radians in the polar image

        The polar image ends at this angle, i.e. the last column of the polar image corresponds to this
        ending angle.

        .. note::
            The polar image does **not** include this angle. It includes all angles starting
            from initial to final angle **excluding** the final angle. Rather, it stops one step size before
            the final angle. Assuming the angular resolution (see :obj:`angleSize`) is small enough, this should not
            matter.

        Radian angle is with respect to the x-axis and rotates counter-clockwise. The angle should be in the range of
        0 to :math:`2\pi`.

        If finalAngle is not set, then it will default to :math:`2\pi`.
    imageSize : (2,) :class:`list`, :class:`tuple` or :class:`numpy.ndarray` of :class:`int`, optional
        Desired size of cartesian image where 1st dimension is number of rows and 2nd dimension is number of columns

        If imageSize is not set, then it defaults to the size required to fit the entire polar image on a cartesian
        image.
    order : :class:`int` (0-5), optional
        The order of the spline interpolation, default is 3. The order has to be in the range 0-5.

        The following orders have special names:

            * 0 - nearest neighbor
            * 1 - bilinear
            * 3 - bicubic
    border : {'constant', 'nearest', 'wrap', 'reflect'}, optional
        Polar points outside the cartesian image boundaries are filled according to the given mode.

        Default is 'constant'

        The following table describes the mode and expected output when seeking past the boundaries. The input column
        is the 1D input array whilst the extended columns on either side of the input array correspond to the expected
        values for the given mode if one extends past the boundaries.

        .. table:: Valid border modes and expected output
            :widths: auto

            ==========  ======  =================  ======
            Mode        Ext.    Input              Ext.
            ==========  ======  =================  ======
            mirror      4 3 2   1 2 3 4 5 6 7 8    7 6 5
            reflect     3 2 1   1 2 3 4 5 6 7 8    8 7 6
            nearest     1 1 1   1 2 3 4 5 6 7 8    8 8 8
            constant    0 0 0   1 2 3 4 5 6 7 8    0 0 0
            wrap        6 7 8   1 2 3 4 5 6 7 8    1 2 3
            ==========  ======  =================  ======

        Refer to :func:`scipy.ndimage.map_coordinates` for more details on this argument.
    borderVal : same datatype as :obj:`image`, optional
        Value used for polar points outside the cartesian image boundaries if :obj:`border` = 'constant'.

        Default is 0.0
    settings : :class:`ImageTransform`, optional
        Contains metadata for conversion between polar and cartesian image.

        Settings contains many of the arguments in :func:`convertToPolarImage` and :func:`convertToCartesianImage` and
        provides an easy way of passing these parameters along without having to specify them all again.

        .. warning::
            Cleaner and more succint to use :meth:`ImageTransform.convertToCartesianImage`

        If settings is not specified, then the other arguments are used in this function and the defaults will be
        calculated if necessary. If settings is given, then the values from settings will be used.

    Returns
    -------
    cartesianImage : (N, M, 3) or (N, M, 4) :class:`numpy.ndarray`
        Cartesian image
    settings : :class:`ImageTransform`
        Contains metadata for conversion between polar and cartesian image.

        Settings contains many of the arguments in :func:`convertToPolarImage` and :func:`convertToCartesianImage` and
        provides an easy way of passing these parameters along without having to specify them all again.
    """
    # Determines whether there are multiple bands or channels in image by checking for 3rd dimension
    isMultiChannel = image.ndim == 3

    if settings is None:
        # Center is set to middle-middle, which means all four quadrants will be shown
        if center is None:
            center = 'middle-middle'

        # Initial radius of the source image
        # In other words, what radius does row 0 correspond to?
        # If not set, default is 0 to get the entire image
        if initialRadius is None:
            initialRadius = 0

        # Final radius of the source image
        # In other words, what radius does the last row of polar image correspond to?
        # If not set, default is the largest radius from image
        if finalRadius is None:
            finalRadius = image.shape[0]

        # Initial angle of the source image
        # In other words, what angle does column 0 correspond to?
        # If not set, default is 0 to get the entire image
        if initialAngle is None:
            initialAngle = 0

        # Final angle of the source image
        # In other words, what angle does the last column of polar image correspond to?
        # If not set, default is 2pi to get the entire image
        if finalAngle is None:
            finalAngle = 2 * np.pi

        # This is used to scale the result of the radius to get the appropriate Cartesian value
        scaleRadius = image.shape[0] / (finalRadius - initialRadius)

        # This is used to scale the result of the angle to get the appropriate Cartesian value
        scaleAngle = image.shape[1] / (finalAngle - initialAngle)

        if imageSize is None:
            # Obtain the image size by looping from initial to final source angle (every possible theta in the image
            # basically)
            thetas = np.mod(np.linspace(0, (finalAngle - initialAngle), image.shape[1]) + initialAngle,
                            2 * np.pi)
            maxRadius = finalRadius * np.ones_like(thetas)

            # Then get the maximum radius of the image and compute the x/y coordinates for each option
            # If a center is not specified, then use the origin as a default. This will be used to determine
            # the new center and image size at once
            if center is not None and not isinstance(center, str):
                xO, yO = getCartesianPoints2(maxRadius, thetas, center)
            else:
                xO, yO = getCartesianPoints2(maxRadius, thetas, np.array([0, 0]))

            # Finally, get the maximum and minimum x/y to obtain the bounds necessary
            # For the minimum x/y, the largest it can be is 0 because of the origin
            # For the maximum x/y, the smallest it can be is 0 because of the origin
            # This happens when the initial and final source angle are in the same quadrant
            # Because of this, it is guaranteed that the min is <= 0 and max is >= 0
            xMin, xMax = min(xO.min(), 0), max(xO.max(), 0)
            yMin, yMax = min(yO.min(), 0), max(yO.max(), 0)

            # Set the image size and center based on the x/y min/max
            if center == 'bottom-left':
                imageSize = np.array([yMax, xMax])
                center = np.array([0, 0])
            elif center == 'bottom-middle':
                imageSize = np.array([yMax, xMax - xMin])
                center = np.array([xMin, 0])
            elif center == 'bottom-right':
                imageSize = np.array([yMax, xMin])
                center = np.array([xMin, 0])
            elif center == 'middle-left':
                imageSize = np.array([yMax - yMin, xMax])
                center = np.array([0, yMin])
            elif center == 'middle-middle':
                imageSize = np.array([yMax - yMin, xMax - xMin])
                center = np.array([xMin, yMin])
            elif center == 'middle-right':
                imageSize = np.array([yMax - yMin, xMin])
                center = np.array([xMin, yMin])
            elif center == 'top-left':
                imageSize = np.array([yMin, xMax])
                center = np.array([0, yMin])
            elif center == 'top-middle':
                imageSize = np.array([yMin, xMax - xMin])
                center = np.array([xMin, yMin])
            elif center == 'top-right':
                imageSize = np.array([yMin, xMin])
                center = np.array([xMin, yMin])

            # When the image size or center are set to x or y min, then that is a negative value
            # Instead of typing abs for each one, an absolute value of the image size and center is done at the end to
            # make it easier.
            imageSize = np.ceil(np.abs(imageSize)).astype(int)
            center = np.ceil(np.abs(center)).astype(int)
        elif isinstance(center, str):
            # Set the center based on the image size given
            if center == 'bottom-left':
                center = imageSize[1::-1] * np.array([0, 0])
            elif center == 'bottom-middle':
                center = imageSize[1::-1] * np.array([1 / 2, 0])
            elif center == 'bottom-right':
                center = imageSize[1::-1] * np.array([1, 0])
            elif center == 'middle-left':
                center = imageSize[1::-1] * np.array([0, 1 / 2])
            elif center == 'middle-middle':
                center = imageSize[1::-1] * np.array([1 / 2, 1 / 2])
            elif center == 'middle-right':
                center = imageSize[1::-1] * np.array([1, 1 / 2])
            elif center == 'top-left':
                center = imageSize[1::-1] * np.array([0, 1])
            elif center == 'top-middle':
                center = imageSize[1::-1] * np.array([1 / 2, 1])
            elif center == 'top-right':
                center = imageSize[1::-1] * np.array([1, 1])

        # Convert image size to tuple to standardize the variable type
        # Some people may use list but we want to convert this
        imageSize = tuple(imageSize)

        settings = ImageTransform(center, initialRadius, finalRadius, initialAngle, finalAngle, imageSize,
                                  image.shape[0:2])
    else:
        # This is used to scale the result of the radius to get the appropriate Cartesian value
        scaleRadius = settings.polarImageSize[0] / (settings.finalRadius - settings.initialRadius)

        # This is used to scale the result of the angle to get the appropriate Cartesian value
        scaleAngle = settings.polarImageSize[1] / (settings.finalAngle - settings.initialAngle)

    # Get list of cartesian x and y coordinate and create a 2D create of the coordinates using meshgrid
    xs = np.arange(0, settings.cartesianImageSize[1])
    ys = np.arange(0, settings.cartesianImageSize[0])
    x, y = np.meshgrid(xs, ys)

    # Take cartesian grid and convert to polar coordinates
    r, theta = getPolarPoints2(x, y, settings.center)

    # Offset the radius by the initial source radius
    r = r - settings.initialRadius

    # Offset the theta angle by the initial source angle
    # The theta values may go past 2pi, so they are looped back around by taking modulo with 2pi.
    # Note: This assumes initial source angle is positive
    theta = np.mod(theta - settings.initialAngle + 2 * np.pi, 2 * np.pi)

    # Scale the radius using scale factor
    r = r * scaleRadius

    # Scale the angle from radians to pixels using scale factor
    theta = theta * scaleAngle

    # Flatten the desired x/y cartesian points into one 2xN array
    desiredCoords = np.vstack((r.flatten(), theta.flatten()))

    # If border is set to constant, then pad the image by the edges by 3 pixels.
    # If one tries to convert back to cartesian without the borders padded then the border of the cartesian image will
    # be corrupted because it will average the pixels with the border value
    if border == 'constant':
        # Pad image by 3 pixels and then offset all of the desired coordinates by 3
        image = np.pad(image, ((3, 3), (3, 3), (0, 0)) if isMultiChannel else 3, 'edge')
        desiredCoords += 3

    # Retrieve cartesian image using map_coordinates. Returns a linear array of the values that
    # must be reshaped into the desired size.
    # For multiple channels, repeat this process for each band and concatenate them at end
    if isMultiChannel:
        cartesianImages = []

        # Assume that there are at least 3 bands in 3D matrix
        for k in range(3):
            cartesianImage = scipy.ndimage.map_coordinates(image[:, :, k], desiredCoords, mode=border, cval=borderVal,
                                                           order=order).reshape(x.shape)
            cartesianImages.append(cartesianImage)

        # If there are 4 bands, then assume the 4th band is alpha
        # We do not want to interpolate the transparency so we just make it all fully opaque
        if image.shape[2] == 4:
            imin, imax = skimage.util.dtype_limits(cartesianImages[0], False)
            cartesianImage = np.full_like(cartesianImages[0], imax)
            cartesianImages.append(cartesianImage)

        cartesianImage = np.dstack(cartesianImages)
    else:
        cartesianImage = scipy.ndimage.map_coordinates(image, desiredCoords, mode=border, cval=borderVal,
                                                       order=order).reshape(x.shape)

    return cartesianImage, settings
