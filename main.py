import polarTransform
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

dataDirectory = os.path.join(os.path.dirname(__file__), 'tests', 'data')

shortAxisApexImage = imageio.imread(os.path.join(dataDirectory, 'shortAxisApex.png'))
shortAxisApexImageGray = shortAxisApexImage[:, :, 0]

polarImage, ptSettings = polarTransform.convertToPolarImage(shortAxisApexImageGray,
                                                            center=np.array([401, 365]))
imageio.imwrite('test.png', np.flipud(polarImage))

# TODO Handle RGB eventually
# TODO Handle rotating 90 degrees
# TODO Check ptSettings for validity
# TODO Clip the radius
# TODO Clip the angle
# TODO Add method support
# Add border support and stuff

plt.imshow(polarImage, cmap='gray', origin='lower')
plt.show()