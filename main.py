import polarTransform
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

dataDirectory = os.path.join(os.path.dirname(__file__), 'tests', 'data')

# TODO Fix issue with ignoregamma, it's weird that it is required
# TODO Fix issue with images being 180 degrees the wrong way
shortAxisApexImage = imageio.imread(os.path.join(dataDirectory, 'shortAxisApex.png'), ignoregamma=True)
# shortAxisApexImage = shortAxisApexImage[:, :, 1]

# polarImage, ptSettings = polarTransform.convertToPolarImage(shortAxisApexImage,
#                                                             center=np.array([401, 365]))
# imageio.imwrite('test.png', np.flipud(polarImage))

# TODO Handle RGB eventually
# TODO Handle rotating 90 degrees
# TODO Check ptSettings for validity
# TODO Clip the radius
# TODO Clip the angle
# TODO Add method support
# TODO Add border support and stuff
# TODO Add note about origin and stuff (should I do that)?

plt.imshow(shortAxisApexImage)#, cmap='gray')
plt.show()