===============
User Guide
===============

.. currentmodule:: polarTransform

:class:`convertToPolarImage` and :class:`convertToCartesianImage` are the two primary classes on which this entire project is based on. The two functions are opposites of one another, reversing the action that the other function does.

Example 1
--------------
Let us take a B-mode echocardiogram and convert it to the polar domain. This is essentially reversing the scan conversion done internally by the ultrasound machine.

Here is the B-mode image:

.. image:: _static/shortAxisApex.png
    :alt: B-mode echocardiogram of short-axis apex view

.. code-block:: python

    import polarTransform
    import matplotlib.pyplot as plt
    import imageio

    cartesianImage = imageio.imread('IMAGE_PATH_HERE')

    polarImage, ptSettings = polarTransform.convertToPolarImage(cartesianImage, center=[401, 365])
    plt.imshow(polarImage, origin='lower')

Resulting polar domain image:

.. image:: _static/shortAxisApexPolarImage.png
    :alt: Polar image of echocardiogram of short-axis apex view

Example 2
--------------
Input image:

.. image:: _static/verticalLines.png
    :alt: Cartesian image

.. code-block:: python

    import polarTransform
    import matplotlib.pyplot as plt
    import imageio

    verticalLinesImage = imageio.imread('IMAGE_PATH_HERE')

    polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLinesImage, initialRadius=30,
                                                                finalRadius=100, initialAngle=2 / 4 * np.pi,
                                                                finalAngle=5 / 4 * np.pi)

    cartesianImage = ptSettings.convertToCartesianImage(polarImage)

    plt.figure()
    plt.imshow(polarImage, origin='lower')

    plt.figure()
    plt.imshow(cartesianImage, origin='lower')

Resulting polar domain image:

.. image:: _static/verticalLinesPolarImage_scaled3.png
    :alt: Polar image

Converting back to the cartesian image results in:

.. image:: _static/verticalLinesCartesianImage_scaled.png
    :alt: Cartesian image