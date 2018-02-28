import polarTransform
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

dataDirectory = os.path.join(os.path.dirname(__file__), 'tests', 'data')

shortAxisApexImage = imageio.imread(os.path.join(dataDirectory, 'shortAxisApex.png'), ignoregamma=True)
shortAxisApexImage = shortAxisApexImage[:, :, 0]

checkerboardImage = imageio.imread(os.path.join(dataDirectory, 'checkerboard.png'), ignoregamma=True)
horizontalLineImage = imageio.imread(os.path.join(dataDirectory, 'horizontalLines.png'), ignoregamma=True)
verticalLineImage = imageio.imread(os.path.join(dataDirectory, 'verticalLines.png'), ignoregamma=True)

# TODO This
# polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLineImage)
# imageio.imwrite('tests\\data\\verticalLinesPolarImage.png', np.flipud(polarImage))

# polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLineImage, initialRadius=30, finalRadius=100)
# imageio.imwrite('tests\\data\\verticalLinesPolarImage_scaled.png', np.flipud(polarImage))
#
# polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLineImage, initialRadius=30, finalRadius=100, initialAngle=2/4 * np.pi, finalAngle=5/4 * np.pi)
# imageio.imwrite('tests\\data\\verticalLinesPolarImage_scaled2.png', np.flipud(polarImage))
#
# polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLineImage, initialRadius=30, finalRadius=100, initialAngle=2/4 * np.pi, finalAngle=5/4 * np.pi, radiusSize=140, angleSize=700)
# imageio.imwrite('tests\\data\\verticalLinesPolarImage_scaled3.png', np.flipud(polarImage))




# polarImage, ptSettings = polarTransform.convertToPolarImage(shortAxisApexImage, center=[401, 365], initialRadius=30, finalRadius=200)
# polarImage, ptSettings = polarTransform.convertToPolarImage(shortAxisApexImage, center=[401, 365], initialRadius=30, finalRadius=200)

# polarImage2, ptSettings2 = polarTransform.convertToPolarImage(np.flipud(shortAxisApexImage),
#                                                             center=np.array([401, 365]), origin='lower')

# TODO Handle RGB eventually
# TODO Handle rotating 90 degrees
# TODO Check ptSettings for validity
# TODO Clip the radius
# TODO Clip the angle
# TODO Add method support
# TODO Add border support and stuff
# TODO Add note about origin and stuff (should I do that)?
# TODO Check origin

# plt.subplot(211)
# plt.imshow(polarImage, cmap='gray', origin='lower')
# plt.subplot(212)
# plt.imshow(shortAxisApexImage, cmap='gray')

# RGB
plt.subplot(211)
plt.imshow(np.abs(compareImage - polarImage), origin='lower')
plt.subplot(212)
plt.imshow(verticalLineImage)
# print(polarImage.shape)

plt.show()