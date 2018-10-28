import os
import sys
import unittest

# Required specifically in each module so that searches happen at the parent directory for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import polarTransform
from polarTransform.tests.util import *


class TestPolarAndCartesianConversion(unittest.TestCase):
    def setUp(self):
        self.shortAxisApexImage = loadImage('shortAxisApex.png')
        self.verticalLinesImage = loadImage('verticalLines.png')

        self.shortAxisApexPolarImage = loadImage('shortAxisApexPolarImage.png')
        self.shortAxisApexPolarImage_centerMiddle = loadImage('shortAxisApexPolarImage_centerMiddle.png')
        self.verticalLinesPolarImage = loadImage('verticalLinesPolarImage.png')
        self.verticalLinesPolarImage_scaled = loadImage('verticalLinesPolarImage_scaled.png')
        self.verticalLinesPolarImage_scaled2 = loadImage('verticalLinesPolarImage_scaled2.png')
        self.verticalLinesPolarImage_scaled3 = loadImage('verticalLinesPolarImage_scaled3.png')

        self.verticalLinesCartesianImage_scaled = loadImage('verticalLinesCartesianImage_scaled.png')
        self.verticalLinesCartesianImage_scaled2 = loadImage('verticalLinesCartesianImage_scaled2.png')
        self.verticalLinesCartesianImage_scaled3 = loadImage('verticalLinesCartesianImage_scaled3.png')

        self.verticalLinesPolarImageBorders = loadImage('verticalLinesPolarImageBorders.png')
        self.verticalLinesCartesianImageBorders2 = loadImage('verticalLinesCartesianImageBorders2.png')
        self.verticalLinesPolarImageBorders3 = loadImage('verticalLinesPolarImageBorders3.png')
        self.verticalLinesCartesianImageBorders4 = loadImage('verticalLinesCartesianImageBorders4.png')

    def test_default(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=np.array([401, 365]))

        cartesianImage = ptSettings.convertToCartesianImage(polarImage)

        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 543)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, (608, 800))
        self.assertEqual(ptSettings.polarImageSize, self.shortAxisApexPolarImage.shape[0:2])

        assert_image_approx_equal_average(cartesianImage, self.shortAxisApexImage, 5)

    def test_default2(self):
        polarImage1, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                     center=np.array([401, 365]), radiusSize=2000,
                                                                     angleSize=4000)

        cartesianImage = ptSettings.convertToCartesianImage(polarImage1)
        ptSettings.polarImageSize = self.shortAxisApexPolarImage.shape[0:2]
        polarImage = ptSettings.convertToPolarImage(cartesianImage)

        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 543)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, (608, 800))
        self.assertEqual(ptSettings.polarImageSize, self.shortAxisApexPolarImage.shape[0:2])

        assert_image_equal(polarImage, self.shortAxisApexPolarImage, 10)

    def test_borders(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, border='constant',
                                                                    borderVal=128.0)

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImageBorders)

        ptSettings.cartesianImageSize = (500, 500)
        ptSettings.center = np.array([250, 250])
        cartesianImage = ptSettings.convertToCartesianImage(polarImage, border='constant', borderVal=255.0)

        np.testing.assert_almost_equal(cartesianImage, self.verticalLinesCartesianImageBorders2)

    def test_borders2(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, border='nearest')

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImageBorders3)

        ptSettings.cartesianImageSize = (500, 500)
        ptSettings.center = np.array([250, 250])
        cartesianImage = ptSettings.convertToCartesianImage(polarImage, border='nearest')

        np.testing.assert_almost_equal(cartesianImage, self.verticalLinesCartesianImageBorders4)


if __name__ == '__main__':
    unittest.main()
