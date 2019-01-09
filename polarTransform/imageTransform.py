class ImageTransform:
    """Class to store settings when converting between cartesian and polar domain"""

    def __init__(self, center, initialRadius, finalRadius, initialAngle, finalAngle, cartesianImageSize,
                 polarImageSize, hasColor):
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
            of 0 to :math:`2\\pi`.
        finalAngle : :class:`float`, optional
            Final angle in radians in the polar image

            The polar image ends at this angle, i.e. the last column of the polar image corresponds to this
            ending angle.

            Radian angle is with respect to the x-axis and rotates counter-clockwise. The angle should be in the range
            of 0 to :math:`2\\pi`.
        cartesianImageSize : (2,) :class:`tuple` of :class:`int`
            Size of cartesian image
        polarImageSize : (2,) :class:`tuple` of :class:`int`
            Size of polar image
        hasColor : :class:`bool`, optional
            Whether or not the polar or cartesian image contains color channels

            This means that the image is structured as (..., y, x, ch) or (..., theta, r, ch) for Cartesian or polar
            images, respectively. If color channels are present, the last dimension (channel axes) will be shifted to
            the front, converted and then shifted back to its original location.

            Default is :obj:`False`

            .. note::
                If an alpha band (4th channel of image is present), then it will be converted. Typically, this is
                unwanted, so the recommended solution is to transform the first 3 channels and set the 4th channel to
                fully on.
        """
        self.center = center
        self.initialRadius = initialRadius
        self.finalRadius = finalRadius
        self.initialAngle = initialAngle
        self.finalAngle = finalAngle
        self.cartesianImageSize = cartesianImageSize
        self.polarImageSize = polarImageSize
        self.hasColor = hasColor

    def convertToPolarImage(self, image, order=3, border='constant', borderVal=0.0, useMultiThreading=False):
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
        image : N-dimensional :class:`numpy.ndarray`
            Cartesian image to convert to polar domain

            Image should be structured in C-order, i.e. the axes should be ordered as (..., z, y, x, [ch]). This format
            is the traditional method of storing image data in Python.

            .. note::
                For multi-dimensional images above 2D, the polar transformation is applied individually across each 2D
                slice. The last two dimensions should be the x & y dimensions, unless :obj:`hasColor` is True in which
                case the 2nd and 3rd to last dimensions should be. The multidimensional shape will be preserved for the
                resulting polar image (besides the Cartesian dimensions).
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
        polarImage : N-dimensional :class:`numpy.ndarray`
            Polar image

            Resulting image is structured in C-order, i.e. the axes are be ordered as (..., z, theta, r, [ch])
            depending on if the input image was 3D. This format is arbitrary but is selected to stay consistent with
            the traditional C-order representation in the Cartesian domain.

            In the mathematical domain, Cartesian
            coordinates are traditionally represented as (x, y, z) and as (r, theta, z) in the polar domain. When
            storing Cartesian data in C-order, the axes are usually flipped and the data is saved as (z, y, x). Thus,
            the polar domain coordinates are also flipped to stay consistent, hence the format (z, theta, r).

            Resulting image shape will be the same as the input image except for the Cartesian dimensions are replaced
            with the polar dimensions.
        """
        image, ptSettings = convertToPolarImage(image, order=order, border=border, borderVal=borderVal,
                                                   useMultiThreading=useMultiThreading, settings=self)
        return image

    def convertToCartesianImage(self, image, order=3, border='constant', borderVal=0.0, useMultiThreading=False):
        """Convert polar image to cartesian image.

        Using a polar image, this function creates a cartesian image. This function is versatile because it can
        automatically calculate an appropiate cartesian image size and center given the polar image. In addition,
        parameters for converting to the polar domain are necessary for the conversion back to the cartesian domain.

        Parameters
        ----------
        image : N-dimensional :class:`numpy.ndarray`
            Polar image to convert to cartesian domain

            Image should be structured in C-order, i.e. the axes should be ordered (..., z, theta, r, [ch]). The channel
            axes should only be present if :obj:`hasColor` is :obj:`True`. This format is arbitrary but is selected to
            stay consistent with the traditional C-order representation in the Cartesian domain.

            In the mathematical domain, Cartesian coordinates are traditionally represented as (x, y, z) and as
            (r, theta, z) in the polar domain. When storing Cartesian data in C-order, the axes are usually flipped and
            the data is saved as (z, y, x). Thus, the polar domain coordinates are also flipped to stay consistent,
            hence the format (z, theta, r).

            .. note::
                For multi-dimensional images above 2D, the cartesian transformation is applied individually across each
                2D slice. The last two dimensions should be the r & theta dimensions, unless :obj:`hasColor` is True in
                which case the 2nd and 3rd to last dimensions should be. The multidimensional shape will be preserved
                for the resulting cartesian image (besides the polar dimensions).
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
        useMultiThreading : :class:`bool`, optional
            Whether to use multithreading when applying transformation for 3D images. This considerably speeds up the
            execution time for large images but adds overhead for smaller 3D images.

            Default is :obj:`False`

        Returns
        -------
        cartesianImage : N-dimensional :class:`numpy.ndarray`
            Cartesian image

            Resulting image is structured in C-order, i.e. the axes are ordered as (..., z, y, x, [ch]). This format is
            the traditional method of storing image data in Python.

            Resulting image shape will be the same as the input image except for the polar dimensions are
            replaced with the Cartesian dimensions.

        See Also
        --------
        :meth:`convertToCartesianImage`
        """
        image, ptSettings = convertToCartesianImage(image, order=order, border=border, borderVal=borderVal,
                                                       useMultiThreading=useMultiThreading, settings=self)
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
               'cartesianImageSize=%s, polarImageSize=%s)' % (self.center, self.initialRadius, self.finalRadius,
                                                              self.initialAngle, self.finalAngle,
                                                              self.cartesianImageSize, self.polarImageSize)

    def __str__(self):
        return self.__repr__()

# Bypasses issue with ImageTransform not being defined for cyclic imports
# The answer is to include imports at the end so that everything is already defined before you import anything else
from polarTransform.convertToCartesianImage import convertToCartesianImage
from polarTransform.convertToPolarImage import convertToPolarImage
from polarTransform.pointsConversion import getCartesianPointsImage, getPolarPointsImage
