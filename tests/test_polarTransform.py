import os
import sys

import imageio
import numpy as np
import skimage

# Look on level up for polarTransform.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import polarTransform
import unittest

dataDirectory = os.path.join(os.path.dirname(__file__), 'data')

def loadImage(filename, flipud=True):
    image = imageio.imread(os.path.join(dataDirectory, filename), ignoregamma=True)
    return np.flipud(image) if flipud else image

class TestPolarConversion(unittest.TestCase):
    def setUp(self):
        self.shortAxisApexImage = imageio.imread(os.path.join(dataDirectory, 'shortAxisApex.png'), ignoregamma=True)
        self.shortAxisApexImage = self.shortAxisApexImage[:, :, 0]

        self.shortAxisApexPolarImage = loadImage('shortAxisApexPolarImage.png')
            # imageio.imread(os.path.join(dataDirectory, 'shortAxisApexPolar.png'),
            #                                           ignoregamma=True)
        # TODO Is this what I want to do?
        # self.shortAxisApexPolarImage = np.flipud(self.shortAxisApexPolarImage)
        self.shortAxisApexPolarImage_centerMiddle = loadImage('shortAxisApexPolarImage_centerMiddle.png')

        # self.shortAxisApexPolarImage_centerMiddle = imageio.imread(
        #     os.path.join(dataDirectory, 'shortAxisApexPolarImage_centerMiddle.png'), ignoregamma=True)
        # self.shortAxisApexPolarImage_centerMiddle = np.flipud(self.shortAxisApexPolarImage_centerMiddle)

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


if __name__ == '__main__':
    unittest.main()
