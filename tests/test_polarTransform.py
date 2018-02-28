import os
import sys

import numpy as np

# Look on level up for polarTransform.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import polarTransform
import unittest

from tests.util import loadImage


class TestPolarConversion(unittest.TestCase):
    def setUp(self):
        self.shortAxisApexImage = loadImage('shortAxisApex.png', False, True)
        self.verticalLinesImage = loadImage('verticalLines.png', False)
        self.horizontalLinesImage = loadImage('horizontalLines.png', False)
        self.checkerboardImage = loadImage('checkerboard.png', False)

        self.shortAxisApexPolarImage = loadImage('shortAxisApexPolarImage.png')
        self.shortAxisApexPolarImage_centerMiddle = loadImage('shortAxisApexPolarImage_centerMiddle.png')
        self.verticalLinesPolarImage = loadImage('verticalLinesPolarImage.png')
        self.verticalLinesPolarImage_scaled = loadImage('verticalLinesPolarImage_scaled.png')
        self.verticalLinesPolarImage_scaled2 = loadImage('verticalLinesPolarImage_scaled2.png')

    # polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLineImage)
    # imageio.imwrite('tests\\data\\verticalLinesPolarImage.png', np.flipud(polarImage))
    #
    # polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLineImage, initialRadius=30, finalRadius=100)
    # imageio.imwrite('tests\\data\\verticalLinesPolarImage_scaled.png', np.flipud(polarImage))
    #
    # polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLineImage, initialRadius=30, finalRadius=100,
    #                                                             initialAngle=2 / 4 * np.pi, finalAngle=5 / 4 * np.pi)
    # imageio.imwrite('tests\\data\\verticalLinesPolarImage_scaled2.png', np.flipud(polarImage))
    #
    # polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLineImage, initialRadius=30, finalRadius=100,
    #                                                             initialAngle=2 / 4 * np.pi, finalAngle=5 / 4 * np.pi,
    #                                                             radiusSize=140, angleSize=700)
    # imageio.imwrite('tests\\data\\verticalLinesPolarImage_scaled3.png', np.flipud(polarImage))

    def test_convertPolarDefault(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=np.array([401, 365]))

        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 543)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.shortAxisApexImage.shape)
        np.testing.assert_array_equal(ptSettings.polarImageSize,
                                      np.array([ptSettings.finalRadius, self.shortAxisApexImage.shape[1] * 2]))
        self.assertEqual(ptSettings.origin, 'upper')

        np.testing.assert_almost_equal(polarImage, self.shortAxisApexPolarImage)

    def test_defaultCenter(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage)

        np.testing.assert_array_equal(ptSettings.center, np.array([400, 304]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 503)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.shortAxisApexImage.shape)
        np.testing.assert_array_equal(ptSettings.polarImageSize,
                                      np.array([ptSettings.finalRadius, self.shortAxisApexImage.shape[1] * 2]))
        self.assertEqual(ptSettings.origin, 'upper')

        np.testing.assert_almost_equal(polarImage, self.shortAxisApexPolarImage_centerMiddle)

    def test_notNumpyArrayCenter(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=[401, 365])
        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        np.testing.assert_almost_equal(polarImage, self.shortAxisApexPolarImage)

        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=(401, 365))
        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        np.testing.assert_almost_equal(polarImage, self.shortAxisApexPolarImage)

    def test_convertPolarRGBA(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 182)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape)
        np.testing.assert_array_equal(ptSettings.polarImageSize,
                                      np.array([ptSettings.finalRadius, self.verticalLinesImage.shape[1] * 2]))
        self.assertEqual(ptSettings.origin, 'upper')

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage)

    def test_convertPolarIFRadius(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, initialRadius=30,
                                                                    finalRadius=100)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape)
        np.testing.assert_array_equal(ptSettings.polarImageSize,
                                      np.array([70, self.verticalLinesImage.shape[1] * 2]))
        self.assertEqual(ptSettings.origin, 'upper')

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage[30:100, :, :])

    def test_convertPolarIFRadiusAngle(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, initialRadius=30,
                                                                    finalRadius=100, initialAngle=2 / 4 * np.pi,
                                                                    finalAngle=5 / 4 * np.pi)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 2 / 4 * np.pi)
        self.assertEqual(ptSettings.finalAngle, 5 / 4 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape)
        np.testing.assert_array_equal(ptSettings.polarImageSize,
                                      np.array([70, 192]))
        self.assertEqual(ptSettings.origin, 'upper')

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage[30:100, 128:320, :])


if __name__ == '__main__':
    unittest.main()
