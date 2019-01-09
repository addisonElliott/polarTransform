import concurrent.futures

import scipy.ndimage


def convertToPolarImage(image, center=None, initialRadius=None, finalRadius=None, initialAngle=None, finalAngle=None,
                        radiusSize=None, angleSize=None, hasColor=False, order=3, border='constant', borderVal=0.0,
                        useMultiThreading=False, settings=None):
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

        Image should be structured in C-order, i.e. the axes should be ordered as (..., z, y, x, [ch]). This format is
        the traditional method of storing image data in Python.

        .. note::
            For multi-dimensional images above 2D, the polar transformation is applied individually across each 2D
            slice. The last two dimensions should be the x & y dimensions, unless :obj:`hasColor` is True in which
            case the 2nd and 3rd to last dimensions should be. The multidimensional shape will be preserved for the
            resulting polar image (besides the Cartesian dimensions).
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
            finalRadius = \\sqrt{((X_{max} - X_{center})^2 + (Y_{max} - Y_{center})^2)}
    initialAngle : :class:`float`, optional
        Starting angle in radians that will appear in the polar image

        The polar image will begin at this angle, i.e. the first column of the polar image will correspond to this
        starting angle.

        Radian angle is with respect to the x-axis and rotates counter-clockwise. The angle should be in the range of
        0 to :math:`2\\pi`.

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
        0 to :math:`2\\pi`.

        If finalAngle is not set, then it will default to :math:`2\\pi`.
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
    hasColor : :class:`bool`, optional
        Whether or not the cartesian image contains color channels

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
            Cleaner and more succint to use :meth:`ImageTransform.convertToPolarImage`

        If settings is not specified, then the other arguments are used in this function and the defaults will be
        calculated if necessary. If settings is given, then the values from settings will be used.

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

        Resulting image shape will be the same as the input image except for the Cartesian dimensions are replaced with
        the polar dimensions.
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
        # If center is not specified, set to the center of the image
        # Image shape is reversed because center is specified as x,y and shape is r,c.
        # Fancy indexing says to grab the last element (x) and to the 2nd to last element (y) and reverse them
        # Otherwise, make sure the center is a Numpy array
        if center is None:
            center = (np.array(image.shape[-1:-3:-1]) / 2).astype(int)
        else:
            center = np.array(center)

        # Initial radius is zero if none is selected
        if initialRadius is None:
            initialRadius = 0

        # Calculate the maximum radius possible
        # Get four corners (indices) of the cartesian image
        # Convert the corners to polar and get the largest radius
        # This will be the maximum radius to represent the entire image in polar
        # For image.shape, grab last 2 elements (y, x). Use -2 in case there is additional dimensions in front
        corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * image.shape[-2:]
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
            cross = np.array([[image.shape[-1] - 1, center[1]], [0, center[1]], [center[0], image.shape[-2] - 1],
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
        settings = ImageTransform(center, initialRadius, finalRadius, initialAngle, finalAngle, image.shape[-2:],
                                  (angleSize, radiusSize), hasColor)

    # Create radii from start to finish with radiusSize, do same for theta
    # Then create a 2D grid of radius and theta using meshgrid
    # Set endpoint to False to NOT include the final sample specified. Think of it like this, if you ask to count from
    # 0 to 30, that is 31 numbers not 30. Thus, we count 0...29 to get 30 numbers.
    radii = np.linspace(settings.initialRadius, settings.finalRadius, settings.polarImageSize[1], endpoint=False)
    theta = np.linspace(settings.initialAngle, settings.finalAngle, settings.polarImageSize[0], endpoint=False)
    r, theta = np.meshgrid(radii, theta)

    # Take polar  grid and convert to cartesian coordinates
    xCartesian, yCartesian = getCartesianPoints2(r, theta, settings.center)

    # Flatten the desired x/y cartesian points into one 2xN array
    desiredCoords = np.vstack((yCartesian.flatten(), xCartesian.flatten()))

    # Get the new shape of the polar image which is the same shape of the cartesian image except the last two dimensions
    # (x & y) are replaced with the polar image size
    newShape = image.shape[:-2] + settings.polarImageSize

    # Reshape the image to be 3D, flattens the array if > 3D otherwise it makes it 3D with the 3rd dimension a size of 1
    image = image.reshape((-1,) + settings.cartesianImageSize)

    # If border is set to constant, then pad the image by the edges by 3 pixels.
    # If one tries to convert back to cartesian without the borders padded then the border of the cartesian image will
    # be corrupted because it will average the pixels with the border value
    if border == 'constant':
        # Pad image by 3 pixels and then offset all of the desired coordinates by 3
        image = np.pad(image, ((0, 0), (3, 3), (3, 3)), 'edge')
        desiredCoords += 3

    if useMultiThreading:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(scipy.ndimage.map_coordinates, slice, desiredCoords, mode=border, cval=borderVal,
                                       order=order) for slice in image]

            concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            polarImages = [future.result().reshape(r.shape) for future in futures]
    else:
        polarImages = []

        # Loop through the third dimension and map each 2D slice
        for slice in image:
            imageSlice = scipy.ndimage.map_coordinates(slice, desiredCoords, mode=border, cval=borderVal,
                                                       order=order).reshape(r.shape)
            polarImages.append(imageSlice)

    # Stack all of the slices together and reshape it to what it should be
    polarImage = np.stack(polarImages, axis=0).reshape(newShape)

    # If there is a color channel present, move it back to the end of axes
    if settings.hasColor:
        polarImage = np.moveaxis(polarImage, 0, -1)

    return polarImage, settings


from polarTransform.imageTransform import ImageTransform
from polarTransform.pointsConversion import *
