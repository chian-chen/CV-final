import cv2
import numpy as np

def detect_lines(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    blurred = cv2.GaussianBlur(blurred, (7, 7), 0)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

    # Perform edge detection
    edged = cv2.Canny(blurred, 50, 100)

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

def evaluation(results):
    GT = [[80, 100, 160, 197],
        [50, 70, 250, 280],
        [95, 115, 345, 375],
        [75, 90, 160, 190],
        [60, 75, 540, 565]]
    
    acc = 0
    for i in range(5):
        pred_open = results[i][0][1]
        pred_close = results[i][1][1]

        if GT[i][0] <= pred_open and GT[i][1] >= pred_open:
            acc += 1
        if GT[i][2] <= pred_close and GT[i][3] >= pred_close:
            acc += 1

    return acc * 10

def event_filter(significant_movement_frames, grouping_threshold, min_event_length):

    events = []
    if significant_movement_frames:
        current_event = [significant_movement_frames[0]]
        for i in range(1, len(significant_movement_frames)):
            if significant_movement_frames[i] - significant_movement_frames[i - 1] <= grouping_threshold:
                current_event.append(significant_movement_frames[i])
            else:
                if len(current_event) >= min_event_length:
                    events.append(current_event)
                current_event = [significant_movement_frames[i]]
        if len(current_event) >= min_event_length:
            events.append(current_event)

    return events
