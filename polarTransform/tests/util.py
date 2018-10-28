import os

import cv2
import imageio
import numpy as np

dataDirectory = os.path.join(os.path.dirname(__file__), 'data')


def loadImage(filename, flipud=True, convertToGrayscale=False):
    image = imageio.imread(os.path.join(dataDirectory, filename), ignoregamma=True)

    if convertToGrayscale:
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
        if convertToGrayscale:
            frame = frame[:, :, 0]
        elif frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[:, :, 0:3]

        # Flip image if desired
        # if flipud:
        #     frame = np.flipud(frame)

        frames.append(frame)

    # Combine the video on the last axis
    image3D = np.stack(frames, axis=-1)

    if flipud:
        image3D = np.flipud(image3D)

    return image3D


def saveImage(filename, image, flipud=True):
    imageio.imwrite(os.path.join(dataDirectory, filename), np.flipud(image) if flipud else image)


def saveVideo(filename, image, flipud=True):
    # Assuming color is present if four dimensions are available
    hasColor = image.ndim == 4

    # Construct codec to use and create writer
    # TODO Unable to set fps to anything besides 0 or OpenCV will use YUV output format
    # causing loss in image data, need upstream fix in OpenCV
    writer = cv2.VideoWriter(os.path.join(dataDirectory, filename), 0, 0, image.shape[1::-1],
                             isColor=hasColor)

    # Flip image if specified
    if flipud:
        image = np.flipud(image)

    # Write frames
    for x in range(image.shape[-1]):
        writer.write(image[..., x])

    writer.release()


def assert_image_equal(desired, actual, diff):
    difference = np.abs(desired.astype(int) - actual.astype(int)).astype(np.uint8)

    assert (np.all(difference <= diff))
