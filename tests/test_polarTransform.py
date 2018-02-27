import os
import sys
import tempfile
import numpy as np
import imageio
import matplotlib.pyplot as plt

# Look on level up for polarTransform.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import polarTransform
import unittest

dataDirectory = os.path.join(os.path.dirname(__file__), 'data')


# DATA_DIR_PATH = os.path.dirname(__file__)
# RAW_NRRD_FILE_PATH = os.path.join(DATA_DIR_PATH, 'BallBinary30x30x30.nrrd')
# RAW_NHDR_FILE_PATH = os.path.join(DATA_DIR_PATH, 'BallBinary30x30x30.nhdr')
# RAW_DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, 'BallBinary30x30x30.raw')
# GZ_NRRD_FILE_PATH = os.path.join(DATA_DIR_PATH, 'BallBinary30x30x30_gz.nrrd')
# BZ2_NRRD_FILE_PATH = os.path.join(DATA_DIR_PATH, 'BallBinary30x30x30_bz2.nrrd')
# GZ_LINESKIP_NRRD_FILE_PATH = os.path.join(DATA_DIR_PATH, 'BallBinary30x30x30_gz_lineskip.nrrd')

class TestPolarConversion(unittest.TestCase):
    def setUp(self):
        self.shortAxisApexImage = imageio.imread(os.path.join(dataDirectory, 'shortAxisApex.png'))
        self.shortAxisApexImage = self.shortAxisApexImage[:, :, 0]

        self.shortAxisApexPolarImage = imageio.imread(os.path.join(dataDirectory, 'shortAxisApexPolar.png'))

    def test_XXX(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=np.array([401, 365]))

        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 543)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.shortAxisApexImage.shape)

        # TODO Check polarImageSize