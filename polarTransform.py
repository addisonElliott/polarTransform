import cv2
import numpy as np
import scipy.interpolate
import scipy.ndimage
import skimage.util


class ImageTransform:
    def __init__(self, center, initialRadius, finalRadius, initialAngle, finalAngle, cartesianImageSize,
                 polarImageSize, origin):
        self.center = center
        self.initialRadius = initialRadius
        self.finalRadius = finalRadius
        self.initialAngle = initialAngle
        self.finalAngle = finalAngle
        self.cartesianImageSize = cartesianImageSize
        self.polarImageSize = polarImageSize
        # Note: Cartesian origin is whether the origin is in upper or lower and only applies for conversion to Cartesian
        # image. The polar image does not matter because (0, 0) should correspond to r=0, theta=0
        self.origin = origin

    def convertToPolarImage(self, image):
        image, ptSettings = convertToPolarImage(image, settings=self)
        return image

    def convertToCartesianImage(self, image):
        image, ptSettings = convertToCartesianImage(image, settings=self)
        return image

    def getPolarPointsImage(self, points):
        return getPolarPointsImage(points, self)

    def getCartesianPointsImage(self, points):
        return getCartesianPointsImage(points, self)

    def __repr__(self):
        return 'ImageTransform(center=%s, initialRadius=%i, finalRadius=%i, initialAngle=%f, finalAngle=%f, ' \
               'cartesianImageSize=%s, polarImageSize=%s)' % (
                   self.center, self.initialRadius, self.finalRadius, self.initialAngle, self.finalAngle,
                   self.cartesianImageSize,
                   self.polarImageSize)

    def __str__(self):
        return self.__repr__()


def getCartesianPoints(rTheta, center):
    if rTheta.ndim == 2:
        x = rTheta[:, 0] * np.cos(rTheta[:, 1]) + center[0]
        y = rTheta[:, 0] * np.sin(rTheta[:, 1]) + center[1]
    else:
        x = rTheta[0] * np.cos(rTheta[1]) + center[0]
        y = rTheta[0] * np.sin(rTheta[1]) + center[1]

    return np.array([x, y]).T


def getCartesianPoints2(r, theta, center):
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]

    return x, y


def getPolarPoints(xy, center):
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
    cX, cY = x - center[0], y - center[1]

    r = np.sqrt(cX ** 2 + cY ** 2)

    theta = np.arctan2(cY, cX)

    # Make range of theta 0 -> 2pi instead of -pi -> pi
    # According to StackOverflow, this is the fastest method:
    # https://stackoverflow.com/questions/37358016/numpy-converting-range-of-angles-from-pi-pi-to-0-2pi
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)

    return r, theta


def getPolarPointsImage(points, settings):
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
                        radiusSize=None, angleSize=None, origin='upper', settings=None):
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

        # Get four corners (indices) of the cartesian image
        # Convert the corners to polar and get the largest radius
        # This will be the maximum radius to represent the entire image in polar
        if finalRadius is None:
            corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * image.shape[0:1]
            radii, _ = getPolarPoints2(corners[:, 1], corners[:, 0], center)
            finalRadius = np.ceil(radii.max()).astype(int)

        # Initial angle of zero if none is selected
        if initialAngle is None:
            initialAngle = 0

        # Final radius is the size of the image so that all points from cartesian are on the polar image
        # Final angle is 2pi to loop throughout entire image
        if finalAngle is None:
            finalAngle = 2 * np.pi

        # If no radius size is given then the size will be the final minus initial radius.
        if radiusSize is None:
            radiusSize = finalRadius - initialRadius

        # Make the angle size be twice the size of the largest dimension of the image
        # This angle size is proportional to the initial and final angle.
        # This was experimentally determined to yield the best resolution
        # The actual answer for the necessary angle size to represent all of the pixels is
        # (finalAngle - initialAngle) / (min(arctan(y / x) - arctan((y - 1) / x)))
        # Where the coordinates used in min are the four corners of the cartesian image with the center
        # subtracted from it. The minimum will be the corner that is the furthest away from the center
        if angleSize is None:
            angleSize = 2 * int(np.max(image.shape) * (finalAngle - initialAngle) / (2 * np.pi))

        # Create the settings
        settings = ImageTransform(center, initialRadius, finalRadius, initialAngle, finalAngle, image.shape,
                                  [radiusSize, angleSize], origin)

    # Flip the image such that the origin is lower left
    # If this does not happen, then 0->2pi rotates clockwise instead of counter-clockwise as is traditional for angles
    if settings.origin == 'upper':
        image = np.flipud(image)

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

    # Retrieve polar image using map_coordinates. Returns a linear array of the values that
    # must be reshaped into the desired size
    # For multiple channels, repeat this process for each band and concatenate them at end
    # Take the transpose of the polar image such that first dimension is radius and second
    # dimension is theta.
    if isMultiChannel:
        polarImages = []

        # Assume that there are at least 3 bands in 3D matrix
        for k in range(3):
            polarImage = scipy.ndimage.map_coordinates(image[:, :, k], desiredCoords, order=3).reshape(r.shape).T
            polarImages.append(polarImage)

        if image.shape[2] == 4:
            imin, imax = skimage.util.dtype_limits(polarImages[0])
            polarImage = np.full_like(polarImages[0], imax)
            polarImages.append(polarImage)

        polarImage = np.dstack(polarImages)
    else:
        polarImage = scipy.ndimage.map_coordinates(image, desiredCoords, order=3).reshape(r.shape).T

    return polarImage, settings


def convertToCartesianImage(image, center=None, initialRadius=None, finalRadius=None, initialSrcRadius=None,
                            finalSrcRadius=None, initialAngle=None, finalAngle=None, initialSrcAngle=None,
                            finalSrcAngle=None, imageSize=None, origin='upper', settings=None):
    if settings is None:
        # Center is set to middle-middle, which means all four quadrants will be shown
        if center is None:
            center = 'middle-middle'

        # Initial radius of the source image
        # In other words, what radius does row 0 correspond to?
        # If not set, default is 0 to get the entire image
        if initialSrcRadius is None:
            initialSrcRadius = 0

        # Final radius of the source image
        # In other words, what radius does the last row of polar image correspond to?
        # If not set, default is the largest radius from image
        if finalSrcRadius is None:
            finalSrcRadius = image.shape[0]

        # Initial angle of the source image
        # In other words, what angle does column 0 correspond to?
        # If not set, default is 0 to get the entire image
        if initialSrcAngle is None:
            initialSrcAngle = 0

        # Final angle of the source image
        # In other words, what angle does the last column of polar image correspond to?
        # If not set, default is 2pi to get the entire image
        if finalSrcAngle is None:
            finalSrcAngle = 2 * np.pi

        # This is used to scale the result of the radius to get the appropriate Cartesian value
        scaleRadius = image.shape[0] / (finalSrcRadius - initialSrcRadius)

        # This is used to scale the result of the angle to get the appropriate Cartesian value
        scaleAngle = image.shape[1] / (finalSrcAngle - initialSrcAngle)

        # This is the size of the cartesian image, should be set somehow...
        if imageSize is None:
            # Obtain the image size by looping from initial to final source angle (every possible theta in the image
            # basically)
            thetas = np.mod(np.linspace(0, (finalSrcAngle - initialSrcAngle), image.shape[1]) + initialSrcAngle,
                            2 * np.pi)
            maxRadius = finalSrcRadius / scaleRadius + initialSrcRadius * np.ones_like(thetas)

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
            imageSize = np.abs(imageSize)
            center = np.abs(center)
        elif isinstance(center, str):
            # Set the center based on the image size given
            if center == 'bottom-left':
                center = imageSize[::-1] * np.array([0, 0])
            elif center == 'bottom-middle':
                center = imageSize[::-1] * np.array([1 / 2, 0])
            elif center == 'bottom-right':
                center = imageSize[::-1] * np.array([1, 1 / 2])
            elif center == 'middle-left':
                center = imageSize[::-1] * np.array([0, 1 / 2])
            elif center == 'middle-middle':
                center = imageSize[::-1] * np.array([1 / 2, 1 / 2])
            elif center == 'middle-right':
                center = imageSize[::-1] * np.array([1, 1])
            elif center == 'top-left':
                center = imageSize[::-1] * np.array([0, 1])
            elif center == 'top-middle':
                center = imageSize[::-1] * np.array([1 / 2, 1])
            elif center == 'top-right':
                center = imageSize[::-1] * np.array([1, 1])

        settings = ImageTransform(center, initialSrcRadius, finalSrcRadius, initialSrcAngle, finalSrcAngle, imageSize,
                                  image.shape, origin)
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

    # Initial angle and final angle set the starting and stopping angle that should be shown on the image
    # Even if the polar image has 0 -> 2*pi, the initial and final angle set the things to select
    # Set all radii that have a theta not in the range of (initialAngle, finalAngle) to be out of bounds
    # This is done by setting the radius to the polar image shape plus initial source radius which is just out
    # of bounds and will be set to the default value
    # TODO Factor in initialAngle/finalAngle into image size and center (ImageTransform, override existing angle)
    if initialAngle is not None and finalAngle is not None:
        r[np.logical_or(theta < initialAngle, theta > finalAngle)] = settings.polarImageSize[0] + settings.initialRadius
    elif initialAngle is not None:
        r[theta < initialAngle] = settings.polarImageSize[0] + settings.initialRadius
    elif finalAngle is not None:
        r[theta > finalAngle] = settings.polarImageSize[0] + settings.initialRadius

    # Initial and final radius set the starting and stopping point that should be shown on the image
    # Set all radii that have a radius less than initial radius or greater than final radius to have a
    # radius out of bounds
    # This is done by setting the radius to the polar image shape plus initial source radius which is just out
    # of bounds and will be set to the default value
    # TODO Factor in initialRadius/finalRadius into image size and center (ImageTransform, override existing radius)
    if initialRadius is not None and finalRadius is not None:
        r[np.logical_or(r < initialRadius, r > finalRadius)] = settings.polarImageSize[0] + settings.initialRadius
    elif initialRadius is not None:
        r[r < initialRadius] = settings.polarImageSize[0] + settings.initialRadius
    elif finalRadius is not None:
        r[r > finalRadius] = settings.polarImageSize[0] + settings.initialRadius

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

    # Convert to 32-bit floats for OpenCV2, does not support double
    r = r.astype(np.float32)
    theta = theta.astype(np.float32)

    # Flatten the desired x/y cartesian points into one 2xN array
    desiredCoords = np.vstack((r.flatten(), theta.flatten()))
    cartesianImage = scipy.ndimage.map_coordinates(image, desiredCoords, order=3).reshape(x.shape)

    # Use OpenCV2 to remap, allows use of their interpolation techniques
    # OpenCV wants x, y coordinates which is opposite of row, col system
    # Theta is the x (col) coordinate, radius is the y (row) coordinate
    # cartesianImage = cv2.remap(image, theta, r, cv2.INTER_CUBIC)

    return cartesianImage, settings
