import os

import cv2
import imageio
import numpy as np

dataDirectory = os.path.join(os.path.dirname(__file__), 'data')


def loadImage(filename, flipud=True, convertToGrayscale=False):
    image = imageio.imread(os.path.join(dataDirectory, filename), ignoregamma=True)

    if convertToGrayscale and image.ndim == 3:
        image = image[:, :, 0]
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, 0:3]

    return np.flipud(image) if flipud else image


def loadVideo(filename, flipud=True, convertToGrayscale=False):
    capture = cv2.VideoCapture(os.path.join(dataDirectory, filename))
    frames = []

    while capture.isOpened():
        # Read frame
        returnValue, frame = capture.read()

        # Error reading image, move on
        if not returnValue:
            break

        # Convert to grayscale or remove alpha component if RGB image
        if convertToGrayscale and frame.ndim == 3:
            frame = frame[:, :, 0]
        elif frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[:, :, 0:3]

        frames.append(frame)

    # Combine the video on the last axis
    image3D = np.stack(frames, axis=-1)

    if flipud:
        image3D = np.flipud(image3D)

    return image3D


def saveImage(filename, image, flipud=True):
    imageio.imwrite(os.path.join(dataDirectory, filename), np.flipud(image) if flipud else image)


def saveVideo(filename, image, fps=20, fourcc='MJLS', flipud=True):
    # Assuming color is present if four dimensions are available
    hasColor = image.ndim == 4

    # Construct four character code
    if isinstance(fourcc, str):
        fourcc = cv2.VideoWriter_fourcc(*fourcc)

    # Construct codec to use and create writer
    # MJLS is one of the few FFMPEG formats that supports lossy encoding in OpenCV specifically because it does not
    # convert to YUV color space
    writer = cv2.VideoWriter(os.path.join(dataDirectory, filename), fourcc, fps, image.shape[1::-1], isColor=hasColor)

    # Flip image if specified
    if flipud:
        image = np.flipud(image)

    # Write frames
    for x in range(image.shape[-1]):
        writer.write(image[..., x])

    # Finish writing
    writer.release()


def assert_image_equal(desired, actual, diff):
    difference = np.abs(desired.astype(int) - actual.astype(int)).astype(np.uint8)

    assert (np.all(difference <= diff))


def assert_image_approx_equal_average(desired, actual, averageDiff):
    assert desired.ndim == actual.ndim, 'Images are not equal, difference in dimensions: %i != %i' % \
                                        (desired.ndim, actual.ndim)
    assert desired.shape == actual.shape, 'Images are not equal, difference in shape %s != %s' % \
                                          (desired.shape, actual.shape)

    # Calculate the difference between the two images
    difference = np.abs(desired.astype(int) - actual.astype(int)).astype(np.uint8)

    # Get average difference between each different pixel
    averageDiffPerPixel = np.sum(difference, axis=(0, 1)) / np.sum(difference > 0, axis=(0, 1))

    assert np.all(averageDiffPerPixel < averageDiff), 'Images are not equal, average difference between each channel ' \
                                                      'is not less than the given threshold, %s < %s' % \
                                                      (averageDiffPerPixel, averageDiff)
