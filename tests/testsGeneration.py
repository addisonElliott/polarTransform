import matplotlib.pyplot as plt
import numpy as np
# from tests.util import loadImage, saveImage
from util import loadImage, saveImage

import polarTransform

shortAxisApexImage = loadImage('shortAxisApex.png')
verticalLinesImage = loadImage('verticalLines.png')
horizontalLinesImage = loadImage('horizontalLines.png')
checkerboardImage = loadImage('checkerboard.png')

shortAxisApexPolarImage = loadImage('shortAxisApexPolarImage.png')
shortAxisApexPolarImage_centerMiddle = loadImage('shortAxisApexPolarImage_centerMiddle.png')
verticalLinesPolarImage = loadImage('verticalLinesPolarImage.png')
verticalLinesPolarImage_scaled = loadImage('verticalLinesPolarImage_scaled.png')
verticalLinesPolarImage_scaled2 = loadImage('verticalLinesPolarImage_scaled2.png')
verticalLinesPolarImage_scaled3 = loadImage('verticalLinesPolarImage_scaled3.png')

verticalLinesCartesianImage_scaled2 = loadImage('verticalLinesCartesianImage_scaled2.png')


def generateShortAxisPolar():
    polarImage, ptSettings = polarTransform.convertToPolarImage(shortAxisApexImage, center=[401, 365])
    saveImage('shortAxisApexPolarImage.png', polarImage)


def generateShortAxisPolar2():
    polarImage, ptSettings = polarTransform.convertToPolarImage(shortAxisApexImage)
    saveImage('shortAxisApexPolarImage_centerMiddle.png', polarImage)


def generateVerticalLinesPolar():
    polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLinesImage)
    saveImage('verticalLinesPolarImage.png', polarImage)


def generateVerticalLinesPolar2():
    polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLinesImage, initialRadius=30, finalRadius=100,
                                                                initialAngle=2 / 4 * np.pi, finalAngle=5 / 4 * np.pi,
                                                                radiusSize=140, angleSize=700)
    saveImage('verticalLinesPolarImage_scaled.png', polarImage)


def generateVerticalLinesPolar3():
    polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLinesImage, initialRadius=30,
                                                                finalRadius=100)
    saveImage('verticalLinesPolarImage_scaled2.png', polarImage)


def generateVerticalLinesPolar4():
    polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLinesImage, initialRadius=30,
                                                                finalRadius=100, initialAngle=2 / 4 * np.pi,
                                                                finalAngle=5 / 4 * np.pi)
    saveImage('verticalLinesPolarImage_scaled3.png', polarImage)


def generateVerticalLinesCartesian():
    cartesianImage, ptSettings = polarTransform.convertToCartesianImage(verticalLinesPolarImage, center=[128, 128],
                                                                        imageSize=[256, 256], finalRadius=182)
    saveImage('verticalLinesCartesianImage.png', cartesianImage)


def generateVerticalLinesCartesian2():
    cartesianImage, ptSettings = polarTransform.convertToCartesianImage(verticalLinesPolarImage_scaled,
                                                                        initialRadius=30, finalRadius=100,
                                                                        initialAngle=2 / 4 * np.pi,
                                                                        finalAngle=5 / 4 * np.pi, imageSize=[256, 256],
                                                                        center=[128, 128])
    saveImage('verticalLinesCartesianImage_scaled.png', cartesianImage)


def generateVerticalLinesCartesian3():
    cartesianImage, ptSettings = polarTransform.convertToCartesianImage(verticalLinesPolarImage_scaled2,
                                                                        center=[128, 128], imageSize=[256, 256],
                                                                        initialRadius=30,
                                                                        finalRadius=100)
    saveImage('verticalLinesCartesianImage_scaled2.png', cartesianImage)


def generateVerticalLinesCartesian4():
    cartesianImage, ptSettings = polarTransform.convertToCartesianImage(verticalLinesPolarImage_scaled3,
                                                                        initialRadius=30,
                                                                        finalRadius=100, initialAngle=2 / 4 * np.pi,
                                                                        finalAngle=5 / 4 * np.pi, center=[128, 128],
                                                                        imageSize=[256, 256])
    saveImage('verticalLinesCartesianImage_scaled3.png', cartesianImage)


def generateShortAxisApexCartesian():
    cartesianImage, ptSettings = polarTransform.convertToCartesianImage(shortAxisApexPolarImage, center=[401, 365],
                                                                        imageSize=[608, 800], finalRadius=543)
    saveImage('shortAxisApexCartesianImage.png', cartesianImage)


def generateShortAxisApexCartesian2():
    cartesianImage, ptSettings = polarTransform.convertToCartesianImage(shortAxisApexPolarImage_centerMiddle,
                                                                        imageSize=[608, 800], finalRadius=503)
    saveImage('shortAxisApexCartesianImage2.png', cartesianImage)

def generateVerticalLinesBorders():
    polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLinesImage, border='constant', borderVal=128.0)
    saveImage('verticalLinesPolarImageBorders.png', polarImage)

    ptSettings.cartesianImageSize = (500, 500)
    ptSettings.center = np.array([250, 250])
    cartesianImage = ptSettings.convertToCartesianImage(polarImage, border='constant', borderVal=255.0)
    saveImage('verticalLinesCartesianImageBorders2.png', cartesianImage)

def generateVerticalLinesBorders2():
    polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLinesImage, border='nearest')
    saveImage('verticalLinesPolarImageBorders3.png', polarImage)

    ptSettings.cartesianImageSize = (500, 500)
    ptSettings.center = np.array([250, 250])
    cartesianImage = ptSettings.convertToCartesianImage(polarImage, border='nearest')
    saveImage('verticalLinesCartesianImageBorders4.png', cartesianImage)


polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLinesImage, order=0)

ptSettings.cartesianImageSize = (500, 500)
ptSettings.center = np.array([250, 250])
cartesianImage = ptSettings.convertToCartesianImage(polarImage)


# polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLinesImage, border='constant', borderVal=128.0)
#
# ptSettings.cartesianImageSize = (500, 500)
# ptSettings.center = np.array([250, 250])
# cartesianImage = ptSettings.convertToCartesianImage(polarImage, border='constant', borderVal=255.0)

# polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLinesImage, border='nearest')
#
# ptSettings.cartesianImageSize = (500, 500)
# ptSettings.center = np.array([250, 250])
# cartesianImage = ptSettings.convertToCartesianImage(polarImage, border='nearest')









# cartesianImage, ptSettings = polarTransform.convertToCartesianImage(verticalLinesPolarImage_scaled2, center=[128, 128], imageSize=[256, 256], initialRadius=30,
#                                                                 finalRadius=100)

# cartesianImage, ptSettings = polarTransform.convertToCartesianImage(verticalLinesPolarImage_scaled3, initialRadius=30,
#                                                             finalRadius=100, initialAngle=2 / 4 * np.pi,
#                                                             finalAngle=5 / 4 * np.pi, center=[128, 128], imageSize=[256, 256])

# cartesianImage, ptSettings = polarTransform.convertToCartesianImage(verticalLinesPolarImage_scaled, initialRadius=30, finalRadius=100,
#                                                             initialAngle=2 / 4 * np.pi, finalAngle=5 / 4 * np.pi, imageSize=[256, 256], center=[128, 128])

# cartesianImage = np.flipud(cartesianImage)
# saveImage('shortAxisApex.png', shortAxisApexImage)
# saveImage('test.png', cartesianImage)
# np.testing.assert_almost_equal(cartesianImage, np.flipud(shortAxisApexImage))
# shortAxisApexImage = np.flipud(shortAxisApexImage)

# diff = verticalLinesImage[:, :, 0:3].astype(int) - np.flipud(cartesianImage[:, :, 0:3]).astype(int)
# diff = np.abs(diff).astype(np.uint8)
# print(diff.dtype, diff.min(), diff.max())
plt.figure()
plt.imshow(verticalLinesImage, cmap='gray', origin='lower')
plt.figure()
plt.imshow(polarImage, cmap='gray', origin='lower')
# plt.figure()
# plt.imshow(verticalLinesCartesianImage_scaled2, origin='lower')
# plt.figure()
# plt.imshow(diff, cmap='gray', origin='upper')

plt.show()

# Enable these functions as you see fit to generate the images
# Note: It is up to the developer to ensure these images are created and look like they are supposed to
# generateShortAxisPolar()
# generateShortAxisPolar2()
# generateVerticalLinesPolar()
# generateVerticalLinesPolar2()
# generateVerticalLinesPolar3()
# generateVerticalLinesPolar4()

# generateVerticalLinesCartesian()
# generateVerticalLinesCartesian2()
# generateVerticalLinesCartesian3()
# generateVerticalLinesCartesian4()

# generateShortAxisApexCartesian()
# generateShortAxisApexCartesian2()

# generateVerticalLinesBorders()
# generateVerticalLinesBorders2()

# TODO Dont forget note that finalRadius/Angle is NOT included. It is everything up to that
# TODO Add method support
# TODO Add note about origin and stuff (should I do that)?
# TODO Check origin
# TODO Add note about angle size and radius size
# TODO Test print(ptSettings)
# TODO Explain order (0-5)
# TODO Add note in docs that cartesianImageSize and polarImageSize only contain first 2 dimensions

# Write order test for polar and cartesian ( just do linear)
# Write settings test for cartesian
# Write tests for polar and cartesian point conversion
