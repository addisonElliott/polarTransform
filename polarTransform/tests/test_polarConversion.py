import os
import sys
import unittest

# Required specifically in each module so that searches happen at the parent directory for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from polarTransform.tests.util import *
import polarTransform


class TestPolarConversion(unittest.TestCase):
    def setUp(self):
        self.shortAxisApexImage = loadImage('shortAxisApex.png')
        self.verticalLinesImage = loadImage('verticalLines.png')

        self.shortAxisApexPolarImage = loadImage('shortAxisApexPolarImage.png')
        self.shortAxisApexPolarImage_centerMiddle = loadImage('shortAxisApexPolarImage_centerMiddle.png')
        self.verticalLinesPolarImage = loadImage('verticalLinesPolarImage.png')
        self.verticalLinesPolarImage_scaled = loadImage('verticalLinesPolarImage_scaled.png')
        self.verticalLinesPolarImage_scaled2 = loadImage('verticalLinesPolarImage_scaled2.png')
        self.verticalLinesPolarImage_scaled3 = loadImage('verticalLinesPolarImage_scaled3.png')

        self.verticalLinesAnimated = loadVideo('verticalLinesAnimated.mp4')
        self.verticalLinesAnimatedPolar = loadVideo('verticalLinesAnimatedPolar.mp4')

    def test_default(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=np.array([401, 365]))

        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 543)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.shortAxisApexImage.shape)
        self.assertEqual(ptSettings.polarImageSize, (802, 1600))

        np.testing.assert_almost_equal(polarImage, self.shortAxisApexPolarImage)

    def test_final_radius(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=np.array([350, 365]))

        np.testing.assert_array_equal(ptSettings.center, np.array([350, 365]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 580)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize,
                         self.shortAxisApexImage.shape)
        self.assertEqual(ptSettings.polarImageSize, (898, 1600))

    def test_defaultCenter(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage)

        np.testing.assert_array_equal(ptSettings.center, np.array([400, 304]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 503)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.shortAxisApexImage.shape)
        self.assertEqual(ptSettings.polarImageSize, (800, 1600))

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
                                                                    finalRadius=100)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape[0:2])
        self.assertEqual(ptSettings.polarImageSize, (99, 1024))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage_scaled2)

    def test_IFRadiusAngle(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, initialRadius=30,
                                                                    finalRadius=100, initialAngle=2 / 4 * np.pi,
                                                                    finalAngle=5 / 4 * np.pi)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 2 / 4 * np.pi)
        self.assertEqual(ptSettings.finalAngle, 5 / 4 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape[0:2])
        self.assertEqual(ptSettings.polarImageSize, (99, 384))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage_scaled3)

    def test_IFRadiusAngleScaled(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, initialRadius=30,
                                                                    finalRadius=100, initialAngle=2 / 4 * np.pi,
                                                                    finalAngle=5 / 4 * np.pi, radiusSize=140,
                                                                    angleSize=700)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 2 / 4 * np.pi)
        self.assertEqual(ptSettings.finalAngle, 5 / 4 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape[0:2])
        self.assertEqual(ptSettings.polarImageSize, (140, 700))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage_scaled)

    def test_settings(self):
        polarImage1, ptSettings1 = polarTransform.convertToPolarImage(self.verticalLinesImage,
                                                                      initialRadius=30, finalRadius=100,
                                                                      initialAngle=2 / 4 * np.pi,
                                                                      finalAngle=5 / 4 * np.pi, radiusSize=140,
                                                                      angleSize=700)

        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, settings=ptSettings1)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 2 / 4 * np.pi)
        self.assertEqual(ptSettings.finalAngle, 5 / 4 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape[0:2])
        self.assertEqual(ptSettings.polarImageSize, (140, 700))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage_scaled)

        polarImage2 = ptSettings1.convertToPolarImage(self.verticalLinesImage)
        np.testing.assert_almost_equal(polarImage2, self.verticalLinesPolarImage_scaled)

    def test_3d_support_rgb(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesAnimated)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 182)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesAnimated.shape[0:2])
        self.assertEqual(ptSettings.polarImageSize, (256, 1024))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesAnimatedPolar)

    def test_3d_supportxxx(self):
        # TODO Do a test where I simply take one image, roll it around, take polar transform and repeat for like some amount of frames.
        # Then compare that to just taking the individual slices and doing it. Probably don't want to do too many frames to not
        # have it slow down.
        #
        # Then repeat the test with a premade MP4 that was generated
        #
        # Probably should do with Bmode (grayscale) and RGB
        # I might want to play around with getting a better grayscale image

        # TODO Load rolled image and then roll it myself...
        # TODO Not sure what I want to do here

        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 182)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape[0:2])
        self.assertEqual(ptSettings.polarImageSize, (256, 1024))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage)
        pass


if __name__ == '__main__':
    unittest.main()
