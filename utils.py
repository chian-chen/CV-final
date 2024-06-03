import cv2
import numpy as np

def detect_lines(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # low_pass filter
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    low_pass = cv2.filter2D(blurred , -1, kernel)

    # Perform edge detection
    edged = cv2.Canny(low_pass, 50, 100)

    # Detect lines using the Hough Line Transform
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

    return lines

def line_pixel_set(lines, shape):
    """
    Generate a set of pixels that represent the lines.
    """
    pixel_set = np.zeros(shape, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(pixel_set, (x1, y1), (x2, y2), 255, 3)
    return pixel_set