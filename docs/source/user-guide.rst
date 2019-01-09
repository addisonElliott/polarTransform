===============
User Guide
===============

.. currentmodule:: polarTransform

:class:`convertToPolarImage` and :class:`convertToCartesianImage` are the two primary functions that make up this package. The two functions are opposites of one another, reversing the action that the other function does.

As the names suggest, the two functions convert an image from the cartesian or polar domain to the other domain with a given set of parameters.  The power of these functions is that the user can specify the resulting image resolution, interpolation order, initial and final radii or angles and much much more. See the :doc:`polarTransform` for more information on the specific parameters that are supported.

Since there are quite a few parameters that can be specified for the conversion functions, the class :class:`ImageTransform` is created and returned from the :class:`convertToPolarImage` or :class:`convertToCartesianImage` functions (along with the converted image) that contains the arguments specified. The benefit of this class is that if one wants to convert the image back to another domain or convert points on either image to/from the other domain, they can simply call the functions within the :class:`ImageTransform` class without specifying all of the arguments again.

The examples below use images from the test suite. The code snippets should run without modification except for changing the paths to point to the correct image.

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
    plt.imshow(polarImage.T, origin='lower')

Resulting polar domain image:

.. image:: _static/shortAxisApexPolarImage.png
    :alt: Polar image of echocardiogram of short-axis apex view

The example input image has a width of 800px and a height of 604px. Since many imaging libraries use C-order rather than Fortran order, the Numpy array containing the image data loaded from imageio has a shape of (604, 800). This is what polarTransform expects for an image where the first dimension is the slowest varying (y) and the last dimension is the fastest varying (x). Additional dimensions can be present before the y & x dimensions in which case polarTransform will transform each 2D slice individually.

The center argument should be a list, tuple or Numpy array of length 2 with format (x, y). A common theme throughout this library is that points will be specified in Fortran order, i.e. (x, y) or (r, theta) whilst data and image sizes will be specified in C-order, i.e. (y, x) or (theta, r).

The polar image returned in this example is in C-order. So this means that the data is (theta, r). When displaying an image using matplotlib, the first dimension is y and second is x. The image is transposed before displaying to flip it 90 degrees.

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
                                                                finalAngle=5 / 4 * np.pi, hasColor=True)

    cartesianImage = ptSettings.convertToCartesianImage(polarImage)

    plt.figure()
    plt.imshow(polarImage.T, origin='lower')

    plt.figure()
    plt.imshow(cartesianImage, origin='lower')

Resulting polar domain image:

.. image:: _static/verticalLinesPolarImage_scaled3.png
    :alt: Polar image

Converting back to the cartesian image results in:

.. image:: _static/verticalLinesCartesianImage_scaled.png
    :alt: Cartesian image

Once again, when displaying polar images using matplotlib, the image is first transposed to rotate the image 90 degrees. This makes it easier to view the image because the theta dimension is longer than the radial dimension.

The hasColor argument was set to True in this example because the image contains color images. The example RGB image has a width and height of 256px. The shape of the image loaded via imageio package is (256, 256, 3). By specified hasColor=True, the last dimension will be shifted to the front and then the polar transformation will occur on each channel separately. Before returning, the function will shift the channel dimension back to the end. If a RGBA image is loaded, it is advised to only transform the first 3 channels and then set the alpha channel to fully on.
