import os
import sys
import unittest

# Required specifically in each module so that searches happen at the parent directory for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import polarTransform
from polarTransform.tests.util import *


class TestPointConversion(unittest.TestCase):
    def setUp(self):
        self.shortAxisApexImage = loadImage('shortAxisApex.png')
        self.shortAxisApexPolarImage = loadImage('shortAxisApexPolarImage.png')

    def test_polarConversion(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=np.array([401, 365]))

        np.testing.assert_array_equal(ptSettings.getPolarPointsImage([401, 365]), np.array([0, 0]))
        np.testing.assert_array_equal(ptSettings.getPolarPointsImage([[401, 365], [401, 365]]),
                                      np.array([[0, 0], [0, 0]]))

        np.testing.assert_array_equal(ptSettings.getPolarPointsImage((401, 365)), np.array([0, 0]))
        np.testing.assert_array_equal(ptSettings.getPolarPointsImage(((401, 365), (401, 365))),
                                      np.array([[0, 0], [0, 0]]))

        np.testing.assert_array_equal(ptSettings.getPolarPointsImage(np.array([401, 365])), np.array([0, 0]))
        np.testing.assert_array_equal(ptSettings.getPolarPointsImage(np.array([[401, 365], [401, 365]])),
                                      np.array([[0, 0], [0, 0]]))

        # Fails here
        np.testing.assert_array_equal(ptSettings.getPolarPointsImage([[451, 365], [401, 400], [348, 365], [401, 305]]),
                                      np.array([[50 * 802 / 543, 0], [35 * 802 / 543, 400], [53 * 802 / 543, 800],
                                                [60 * 802 / 543, 1200]]))

    def test_cartesianConversion(self):
        cartesianImage, ptSettings = polarTransform.convertToCartesianImage(self.shortAxisApexPolarImage,
                                                                            center=[401, 365], imageSize=[608, 800],
                                                                            finalRadius=543)

        np.testing.assert_array_equal(ptSettings.getCartesianPointsImage([0, 0]), np.array([401, 365]))
        np.testing.assert_array_equal(ptSettings.getCartesianPointsImage([[0, 0], [0, 0]]),
                                      np.array([[401, 365], [401, 365]]))

        np.testing.assert_array_equal(ptSettings.getCartesianPointsImage((0, 0)), np.array([401, 365]))
        np.testing.assert_array_equal(ptSettings.getCartesianPointsImage(((0, 0), (0, 0))),
                                      np.array([[401, 365], [401, 365]]))

        np.testing.assert_array_equal(ptSettings.getCartesianPointsImage(np.array([0, 0])), np.array([401, 365]))
        np.testing.assert_array_equal(ptSettings.getCartesianPointsImage(np.array([[0, 0], [0, 0]])),
                                      np.array([[401, 365], [401, 365]]))

        np.testing.assert_array_equal(ptSettings.getCartesianPointsImage(
            np.array([[50 * 802 / 543, 0], [35 * 802 / 543, 400], [53 * 802 / 543, 800],
                      [60 * 802 / 543, 1200]])), np.array([[451, 365], [401, 400], [348, 365], [401, 305]]))


if __name__ == '__main__':
    unittest.main()
