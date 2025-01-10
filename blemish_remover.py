import cv2
import numpy as np

WINDOW_NAME = "Blemish Remover"
DEFAULT_RADIUS = 15
MAX_SOBEL_VALUE = 1000.0
NEIGHBORHOOD_SQUARES = 3


def calculate_sobel_mean(square_block):
    """
    Calculates the mean of the Sobel derivatives in both X and Y directions
    for a given square block. This provides a measure of the derivative
    or roughness of the square block, which can be helpful in edge-detection
    tasks or assessing texture levels within a localized region of the image.

    :param square_block: A 2D array representing the square image block
        on which the Sobel derivative mean will be calculated.
        It is assumed to be a grayscale image where Sobel operations
        can be applied for edge detection or roughness evaluation.

    :return: The total mean of the absolute Sobel derivatives computed
        in the X and Y directions for the square block, reflecting
        the overall roughness of the image block.
    """
    #Find a measure of derivative/roughness of the square block
    #Calculate the mean of Sobel derivatives in X and Y directions.
    mean_sobel_x = np.mean(np.abs(cv2.Sobel(square_block, cv2.CV_32F, 1, 0)))
    mean_sobel_y = np.mean(np.abs(cv2.Sobel(square_block, cv2.CV_32F, 0, 1)))
    return mean_sobel_x + mean_sobel_y


def is_within_image_bounds(start, size, img_shape):
    """
    Check if a square region defined by a starting coordinate
    and size lies completely within the bounds of a given image shape.

    This function determines if the square, starting from the given
    top-left coordinate and extending equally along width and height
    based on the specified size, does not exceed the image's boundaries.

    :param start: A tuple (x, y) representing the top-left corner of the square.
    :param size: An integer specifying the width and height of the square.
    :param img_shape: A tuple (height, width) specifying the dimensions of the image.
    :return: A boolean value indicating whether the square lies fully within
        the image boundaries.
    """
    # Check if the square region lies within the image bounds.
    return (start[0] >= 0 and start[1] >= 0 and
            start[0] + size <= img_shape[1] and
            start[1] + size <= img_shape[0])


def find_smoothest_region(img, center):
    """
    Find the smoothest region around the given center in an image.

    The function searches for the region with the lowest mean Sobel value
    within a specified neighborhood around the provided center. Sobel
    values are used as a measure of edge intensity, and the smoothest
    region corresponds to the area with the minimum edge intensity. The
    search area is divided into a grid of squares, and the smoothness is
    calculated for each square to determine the smoothest one.

    :param img: The input image as a numpy array. Expected to be of shape
                (height, width, channels).
    :type img: numpy.ndarray
    :param center: A tuple representing the x, y coordinates of the center
                   around which to search for the smoothest region.
    :type center: tuple[int, int]
    :return: The sub-array representing the smoothest region found within
             the specified neighborhood, or None if no valid region was
             found.
    :rtype: numpy.ndarray | None
    """
    # Find the smoothest region around the given center.
    min_sobel_val = MAX_SOBEL_VALUE
    start_position = np.array(center, dtype=int) - NEIGHBORHOOD_SQUARES * DEFAULT_RADIUS
    square_size = 2 * DEFAULT_RADIUS
    best_square = None

    for i in range(NEIGHBORHOOD_SQUARES):
        for j in range(NEIGHBORHOOD_SQUARES):
            new_position = start_position + np.array([i * square_size, j * square_size])
            if not is_within_image_bounds(new_position, square_size, img.shape):
                continue

            square_block = img[new_position[1]:new_position[1] + square_size,
                           new_position[0]:new_position[0] + square_size, :]
            sobel_value = calculate_sobel_mean(square_block)

            if sobel_value < min_sobel_val:
                min_sobel_val = sobel_value
                best_square = square_block

    return best_square


def on_mouse_event(event, x, y, flags, param):
    """
    Handles mouse events for selecting and repairing blemishes on an image. This function processes
    left mouse button clicks to remove blemishes by cloning smooth regions over the blemish. Right
    mouse button clicks are used to undo the last blemish removal. It leverages OpenCV for image
    processing and display updates.

    :param event: The mouse event triggered, such as left button click or right button click, from
                  OpenCV mouse event constants.
    :type event: int
    :param x: The X-coordinate of the mouse click position within the image.
    :type x: int
    :param y: The Y-coordinate of the mouse click position within the image.
    :type y: int
    :param flags: Any relevant flags passed by OpenCV during the mouse callback event, such as
                  modifier keys.
    :type flags: int
    :param param: Additional parameters passed to the callback, if any.
    :type param: Any
    :return: None
    """
    # Handle mouse events for selecting and repairing blemishes.

    global image, blemish_region, mask, center

    if event == cv2.EVENT_LBUTTONDOWN:  # Left click for blemish removal
        center = (x, y)
        if not is_within_image_bounds((x - DEFAULT_RADIUS, y - DEFAULT_RADIUS), 2 * DEFAULT_RADIUS, image.shape):
            return

        blemish_region = image[y - DEFAULT_RADIUS:y + DEFAULT_RADIUS, x - DEFAULT_RADIUS:x + DEFAULT_RADIUS, :]
        smooth_region = find_smoothest_region(image, center)

        mask = np.zeros(smooth_region.shape, smooth_region.dtype)
        cv2.circle(mask, (DEFAULT_RADIUS, DEFAULT_RADIUS), DEFAULT_RADIUS, (255, 255, 255), thickness=-1,
                   lineType=cv2.LINE_AA)

        image = cv2.seamlessClone(smooth_region, image, mask, center, cv2.NORMAL_CLONE)
        cv2.imshow(WINDOW_NAME, image)

    elif event == cv2.EVENT_RBUTTONDOWN:  # Right click for undo
        image = cv2.seamlessClone(blemish_region, image, mask, center, cv2.NORMAL_CLONE)
        cv2.imshow(WINDOW_NAME, image)


# load blemish image
filename = "blemish.png"
image = cv2.imread(filename)

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse_event)

while True:
    cv2.imshow(WINDOW_NAME, image)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key to exit
        cv2.destroyAllWindows()
        break

    if key == ord("s"):  # Save the image if 's' is pressed
        cv2.imwrite('blemish-removed.jpg', image)
