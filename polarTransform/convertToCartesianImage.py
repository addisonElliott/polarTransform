import concurrent.futures

import scipy.ndimage


def convertToCartesianImage(image, center=None, initialRadius=None,
                            finalRadius=None, initialAngle=None,
                            finalAngle=None, imageSize=None, hasColor=False, order=3, border='constant',
                            borderVal=0.0, useMultiThreading=False, settings=None):
    """Convert polar image to cartesian image.

    Using a polar image, this function creates a cartesian image. This function is versatile because it can
    automatically calculate an appropriate cartesian image size and center given the polar image. In addition,
    parameters for converting to the polar domain are necessary for the conversion back to the cartesian domain.

    Parameters
    ----------
    image : N-dimensional :class:`numpy.ndarray`
        Polar image to convert to cartesian domain

        Image should be structured in C-order, i.e. the axes should be ordered (..., z, theta, r, [ch]). The channel
        axes should only be present if :obj:`hasColor` is :obj:`True`. This format is arbitrary but is selected to stay
        consistent with the traditional C-order representation in the Cartesian domain.

        In the mathematical domain, Cartesian coordinates are traditionally represented as (x, y, z) and as
        (r, theta, z) in the polar domain. When storing Cartesian data in C-order, the axes are usually flipped and the
        data is saved as (z, y, x). Thus, the polar domain coordinates are also flipped to stay consistent, hence the
        format (z, theta, r).

        .. note::
            For multi-dimensional images above 2D, the cartesian transformation is applied individually across each
            2D slice. The last two dimensions should be the r & theta dimensions, unless :obj:`hasColor` is True in
            which case the 2nd and 3rd to last dimensions should be. The multidimensional shape will be preserved
            for the resulting cartesian image (besides the polar dimensions).
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
        0 to :math:`2\\pi`.

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
        0 to :math:`2\\pi`.

        If finalAngle is not set, then it will default to :math:`2\\pi`.
    imageSize : (2,) :class:`list`, :class:`tuple` or :class:`numpy.ndarray` of :class:`int`, optional
        Desired size of cartesian image where 1st dimension is number of rows and 2nd dimension is number of columns

        If imageSize is not set, then it defaults to the size required to fit the entire polar image on a cartesian
        image.
    hasColor : :class:`bool`, optional
        Whether or not the polar image contains color channels

        This means that the image is structured as (..., y, x, ch) or (..., theta, r, ch) for Cartesian or polar
        images, respectively. If color channels are present, the last dimension (channel axes) will be shifted to
        the front, converted and then shifted back to its original location.

        Default is :obj:`False`

        .. note::
            If an alpha band (4th channel of image is present), then it will be converted. Typically, this is
            unwanted, so the recommended solution is to transform the first 3 channels and set the 4th channel to
            fully on.
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
    useMultiThreading : :class:`bool`, optional
        Whether to use multithreading when applying transformation for 3D images. This considerably speeds up the
        execution time for large images but adds overhead for smaller 3D images.

        Default is :obj:`False`
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
    cartesianImage : N-dimensional :class:`numpy.ndarray`
            Cartesian image

            Resulting image is structured in C-order, i.e. the axes are ordered as (..., z, y, x, [ch]). This format is
            the traditional method of storing image data in Python.

            Resulting image shape will be the same as the input image except for the polar dimensions are
            replaced with the Cartesian dimensions.
    settings : :class:`ImageTransform`
        Contains metadata for conversion between polar and cartesian image.

        Settings contains many of the arguments in :func:`convertToPolarImage` and :func:`convertToCartesianImage` and
        provides an easy way of passing these parameters along without having to specify them all again.
    """

    # If there is a color channel present, move it to the front of axes
    if settings.hasColor if settings is not None else hasColor:
        image = np.moveaxis(image, -1, 0)

    # Create settings if none are given
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
            finalRadius = image.shape[-1]

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
        scaleRadius = image.shape[-1] / (finalRadius - initialRadius)

        # This is used to scale the result of the angle to get the appropriate Cartesian value
        scaleAngle = image.shape[-2] / (finalAngle - initialAngle)

        if imageSize is None:
            # Obtain the image size by looping from initial to final source angle (every possible theta in the image
            # basically)
            thetas = np.mod(np.linspace(0, (finalAngle - initialAngle), image.shape[-2]) + initialAngle, 2 * np.pi)
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
                                  image.shape[-2:], hasColor)
    else:
        # This is used to scale the result of the radius to get the appropriate Cartesian value
        scaleRadius = settings.polarImageSize[1] / (settings.finalRadius - settings.initialRadius)

        # This is used to scale the result of the angle to get the appropriate Cartesian value
        scaleAngle = settings.polarImageSize[0] / (settings.finalAngle - settings.initialAngle)

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
    desiredCoords = np.vstack((theta.flatten(), r.flatten()))

    # Get the new shape which is the cartesian image shape plus any other dimensions
    # Get the new shape of the cartesian image which is the same shape of the polar image except the last two dimensions
    # (r & theta) are replaced with the cartesian image size
    newShape = image.shape[:-2] + settings.cartesianImageSize

    # Reshape the image to be 3D, flattens the array if > 3D otherwise it makes it 3D with the 3rd dimension a size of 1
    image = image.reshape((-1,) + settings.polarImageSize)

    if border == 'constant':
        # Pad image by 3 pixels and then offset all of the desired coordinates by 3
        image = np.pad(image, ((0, 0), (3, 3), (3, 3)), 'edge')
        desiredCoords += 3

    if useMultiThreading:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(scipy.ndimage.map_coordinates, slice, desiredCoords, mode=border, cval=borderVal,
                                       order=order) for slice in image]

            concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            cartesianImages = [future.result().reshape(x.shape) for future in futures]
    else:
        cartesianImages = []

        # Loop through the third dimension and map each 2D slice
        for slice in image:
            imageSlice = scipy.ndimage.map_coordinates(slice, desiredCoords, mode=border, cval=borderVal,
                                                       order=order).reshape(x.shape)
            cartesianImages.append(imageSlice)

    # Stack all of the slices together and reshape it to what it should be
    cartesianImage = np.stack(cartesianImages, axis=0).reshape(newShape)

    # If there is a color channel present, move it abck to the end of axes
    if settings.hasColor:
        cartesianImage = np.moveaxis(cartesianImage, 0, -1)

    return cartesianImage, settings


from polarTransform.pointsConversion import *
from polarTransform.imageTransform import ImageTransform
