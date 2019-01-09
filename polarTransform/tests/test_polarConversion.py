import os
import sys
import unittest

# Required specifically in each module so that searches happen at the parent directory for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import polarTransform
from polarTransform.tests.util import *


class TestPolarConversion(unittest.TestCase):
    def setUp(self):
        self.shortAxisApexImage = loadImage('shortAxisApex.png')
        self.verticalLinesImage = loadImage('verticalLines.png')
        self.horizontalLinesImage = loadImage('horizontalLines.png', convertToGrayscale=True)

        self.shortAxisApexPolarImage = loadImage('shortAxisApexPolarImage.png')
        self.shortAxisApexPolarImage_centerMiddle = loadImage('shortAxisApexPolarImage_centerMiddle.png')
        self.verticalLinesPolarImage = loadImage('verticalLinesPolarImage.png')
        self.verticalLinesPolarImage_scaled = loadImage('verticalLinesPolarImage_scaled.png')
        self.verticalLinesPolarImage_scaled2 = loadImage('verticalLinesPolarImage_scaled2.png')
        self.verticalLinesPolarImage_scaled3 = loadImage('verticalLinesPolarImage_scaled3.png')

        self.horizontalLinesPolarImage = loadImage('horizontalLinesPolarImage.png', convertToGrayscale=True)

        self.verticalLinesAnimated = loadVideo('verticalLinesAnimated.avi')
        self.verticalLinesAnimatedPolar = loadVideo('verticalLinesAnimatedPolar.avi')
        self.horizontalLinesAnimated = loadVideo('horizontalLinesAnimated.avi', convertToGrayscale=True)
        self.horizontalLinesAnimatedPolar = loadVideo('horizontalLinesAnimatedPolar.avi', convertToGrayscale=True)

    def test_default(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=np.array([401, 365]))

        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 543)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.shortAxisApexImage.shape)
        self.assertEqual(ptSettings.polarImageSize, (1600, 802))

        np.testing.assert_almost_equal(polarImage, self.shortAxisApexPolarImage)

    def test_final_radius(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=np.array([350, 365]))

        np.testing.assert_array_equal(ptSettings.center, np.array([350, 365]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 580)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.shortAxisApexImage.shape)
        self.assertEqual(ptSettings.polarImageSize, (1600, 898))

    def test_defaultCenter(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage)

        np.testing.assert_array_equal(ptSettings.center, np.array([400, 304]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 503)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.shortAxisApexImage.shape)
        self.assertEqual(ptSettings.polarImageSize, (1600, 800))

        np.testing.assert_almost_equal(polarImage, self.shortAxisApexPolarImage_centerMiddle)

    def test_notNumpyArrayCenter(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage, center=(401, 365))
        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        np.testing.assert_almost_equal(polarImage, self.shortAxisApexPolarImage)

        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage, center=(401, 365))
        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        np.testing.assert_almost_equal(polarImage, self.shortAxisApexPolarImage)

    def test_IFRadius(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, initialRadius=30,
                                                                    finalRadius=100, hasColor=True)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape[-3:-1])
        self.assertEqual(ptSettings.polarImageSize, (1024, 99))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage_scaled2)

    def test_IFRadiusAngle(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, initialRadius=30,
                                                                    finalRadius=100, initialAngle=2 / 4 * np.pi,
                                                                    finalAngle=5 / 4 * np.pi, hasColor=True)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 2 / 4 * np.pi)
        self.assertEqual(ptSettings.finalAngle, 5 / 4 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape[-3:-1])
        self.assertEqual(ptSettings.polarImageSize, (384, 99))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage_scaled3)

    def test_IFRadiusAngleScaled(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, initialRadius=30,
                                                                    finalRadius=100, initialAngle=2 / 4 * np.pi,
                                                                    finalAngle=5 / 4 * np.pi, radiusSize=140,
                                                                    angleSize=700, hasColor=True)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 2 / 4 * np.pi)
        self.assertEqual(ptSettings.finalAngle, 5 / 4 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape[-3:-1])
        self.assertEqual(ptSettings.polarImageSize, (700, 140))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage_scaled)

    def test_settings(self):
        polarImage1, ptSettings1 = polarTransform.convertToPolarImage(self.verticalLinesImage,
                                                                      initialRadius=30, finalRadius=100,
                                                                      initialAngle=2 / 4 * np.pi,
                                                                      finalAngle=5 / 4 * np.pi, radiusSize=140,
                                                                      angleSize=700, hasColor=True)

        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, settings=ptSettings1)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 2 / 4 * np.pi)
        self.assertEqual(ptSettings.finalAngle, 5 / 4 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape[-3:-1])
        self.assertEqual(ptSettings.polarImageSize, (700, 140))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage_scaled)

        polarImage2 = ptSettings1.convertToPolarImage(self.verticalLinesImage)
        np.testing.assert_almost_equal(polarImage2, self.verticalLinesPolarImage_scaled)

    def test_default_horizontalLines(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.horizontalLinesImage)

        np.testing.assert_array_equal(ptSettings.center, np.array([512, 384]))
        self.assertEqual(ptSettings.initialRadius, 0)
        # sqrt(512^2 + 384^2) maximum distance = 640
        self.assertEqual(ptSettings.finalRadius, 640)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.horizontalLinesImage.shape)
        self.assertEqual(ptSettings.polarImageSize, (2048, 1024))

        np.testing.assert_almost_equal(polarImage, self.horizontalLinesPolarImage)

    def test_3d_support_rgb(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesAnimated, hasColor=True)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 182)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesAnimated.shape[-3:-1])
        self.assertEqual(ptSettings.polarImageSize, (1024, 256))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesAnimatedPolar)

    def test_3d_support_rgb_multithreaded(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesAnimated, hasColor=True,
                                                                    useMultiThreading=True)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 182)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesAnimated.shape[-3:-1])
        self.assertEqual(ptSettings.polarImageSize, (1024, 256))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesAnimatedPolar)

    def test_3d_support_grayscale(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.horizontalLinesAnimated)

        np.testing.assert_array_equal(ptSettings.center, np.array([512, 384]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 640)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.horizontalLinesAnimated.shape[-2:])
        self.assertEqual(ptSettings.polarImageSize, (2048, 1024))

        np.testing.assert_almost_equal(polarImage, self.horizontalLinesAnimatedPolar)

    def test_3d_support_grayscale_multithreaded(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.horizontalLinesAnimated,
                                                                    useMultiThreading=True)

        np.testing.assert_array_equal(ptSettings.center, np.array([512, 384]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 640)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.horizontalLinesAnimated.shape[-2:])
        self.assertEqual(ptSettings.polarImageSize, (2048, 1024))

        np.testing.assert_almost_equal(polarImage, self.horizontalLinesAnimatedPolar)


if __name__ == '__main__':
    unittest.main()
