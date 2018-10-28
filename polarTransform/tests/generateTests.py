import os
import sys

# Required specifically in each module so that searches happen at the parent directory for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import polarTransform
from polarTransform.tests.util import *

# This file should only be ran to generate images contained in the data folder. Since much of this library is performing
# actions on images, the best way to validate through tests that it is working correctly, is to generate images using
# the library and then visually inspecting that the image looks correct.
#
# These images are uploaded and are apart of the repository itself so most of these images will not need to be
# regenerated unless a breaking change is made to the code that changes the output.

# Load input images that are used to generate output images
shortAxisApexImage = loadImage('shortAxisApex.png')
verticalLinesImage = loadImage('verticalLines.png')
horizontalLines = loadImage('horizontalLines.png', convertToGrayscale=True)

shortAxisApexPolarImage = loadImage('shortAxisApexPolarImage.png')
shortAxisApexPolarImage_centerMiddle = loadImage('shortAxisApexPolarImage_centerMiddle.png')
verticalLinesPolarImage = loadImage('verticalLinesPolarImage.png')
verticalLinesPolarImage_scaled = loadImage('verticalLinesPolarImage_scaled.png')
verticalLinesPolarImage_scaled2 = loadImage('verticalLinesPolarImage_scaled2.png')
verticalLinesPolarImage_scaled3 = loadImage('verticalLinesPolarImage_scaled3.png')

verticalLinesCartesianImage_scaled2 = loadImage('verticalLinesCartesianImage_scaled2.png')

verticalLinesAnimated = loadVideo('verticalLinesAnimated.avi')
horizontalLinesAnimated = loadVideo('horizontalLinesAnimated.avi')


# Generate functions
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
                                                                        initialRadius=30, finalRadius=100)
    saveImage('verticalLinesCartesianImage_scaled2.png', cartesianImage)


def generateVerticalLinesCartesian4():
    cartesianImage, ptSettings = polarTransform.convertToCartesianImage(verticalLinesPolarImage_scaled3,
                                                                        initialRadius=30, finalRadius=100,
                                                                        initialAngle=2 / 4 * np.pi,
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


def generateHorizontalLinesPolar():
    polarImage, ptSettings = polarTransform.convertToPolarImage(horizontalLines)
    saveImage('horizontalLinesPolarImage.png', polarImage)


def generateVerticalLinesAnimated():
    frameSize = 40

    frames = [np.roll(verticalLinesImage, 12 * x, axis=1) for x in range(frameSize)]
    image3D = np.stack(frames, axis=-1)

    saveVideo('verticalLinesAnimated.avi', image3D)


def generateVerticalLinesAnimatedPolar():
    frameSize = 40
    ptSettings = None
    polarFrames = []

    for x in range(frameSize):
        frame = verticalLinesAnimated[..., x]

        # Call convert to polar image on each frame, uses the assumption that individual 2D image works fine based on
        # other tests
        if ptSettings:
            polarFrame = ptSettings.convertToPolarImage(frame)
        else:
            polarFrame, ptSettings = polarTransform.convertToPolarImage(frame)

        polarFrames.append(polarFrame)

    polarImage3D = np.stack(polarFrames, axis=-1)
    saveVideo('verticalLinesAnimatedPolar.avi', polarImage3D)


def generateHorizontalLinesAnimated():
    frameSize = 40

    frames = [np.roll(horizontalLines, 36 * x, axis=0) for x in range(frameSize)]
    image3D = np.stack(frames, axis=-1)

    saveVideo('horizontalLinesAnimated.avi', image3D)


def generateHorizontalLinesAnimatedPolar():
    frameSize = 40
    ptSettings = None
    polarFrames = []

    for x in range(frameSize):
        frame = horizontalLinesAnimated[..., x]

        # Call convert to polar image on each frame, uses the assumption that individual 2D image works fine based on
        # other tests
        if ptSettings:
            polarFrame = ptSettings.convertToPolarImage(frame)
        else:
            polarFrame, ptSettings = polarTransform.convertToPolarImage(frame)

        polarFrames.append(polarFrame)

    polarImage3D = np.stack(polarFrames, axis=-1)
    saveVideo('horizontalLinesAnimatedPolar.avi', polarImage3D)


# Enable these functions as you see fit to generate the images
# Note: It is up to the developer to visually inspect the output images that are created.
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

# generateHorizontalLinesPolar()

# generateVerticalLinesAnimated()
# generateVerticalLinesAnimatedPolar()

# generateHorizontalLinesAnimated()
# generateHorizontalLinesAnimatedPolar()

# TODO Fix Travis CI so that it doesnt run on all branches...? NEED TO TEST IT NOW
# TODO Further test and refine the ImageTransform class, know I messed it up somehow
# TODO Setup automatic release like pynrrd NEED TO TEST IT NOW
# TODO Change how tests are run in travis and stuff NEED TO TEST IT NOW

# TODO Look at tests and see if I cant get rid of the numbered ones that look the same

# TODO Will need to install dev-requirements.txt for TravisCI
# TODO Enable codecov for Travis CI, add banner for it