import polarTransform
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

dataDirectory = os.path.join(os.path.dirname(__file__), 'tests', 'data')

shortAxisApexImage = imageio.imread(os.path.join(dataDirectory, 'shortAxisApex.png'), ignoregamma=True)
shortAxisApexImage = shortAxisApexImage[:, :, 0]

polarImage, ptSettings = polarTransform.convertToPolarImage(shortAxisApexImage,
                                                            center=np.array([401, 365]))

polarImage2, ptSettings2 = polarTransform.convertToPolarImage(np.flipud(shortAxisApexImage),
                                                            center=np.array([401, 365]), origin='lower')
# imageio.imwrite('test.png', np.flipud(polarImage))

# TODO Handle RGB eventually
# TODO Handle rotating 90 degrees
# TODO Check ptSettings for validity
# TODO Clip the radius
# TODO Clip the angle
# TODO Add method support
# TODO Add border support and stuff
# TODO Add note about origin and stuff (should I do that)?
# TODO Check origin

plt.subplot(211)
plt.imshow(polarImage, cmap='gray')
plt.subplot(212)
plt.imshow(polarImage2, cmap='gray')
plt.show()