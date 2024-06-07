import cv2
import numpy as np
import time

def detect_lines(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # blurred = cv2.GaussianBlur(blurred, (7, 7), 0)
    # blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    # blurred = cv2.medianBlur(blurred, 15)
    erosion = cv2.erode(blurred, kernel=np.ones((3, 3), np.uint8), iterations=3)

    # Perform edge detection
    edged = cv2.Canny(erosion, 50, 100)
    # edged = cv2.Canny(blurred, 50, 100)

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

def detection(video_path, significant_movement_threshold, grouping_threshold, min_event_length):
    cap = cv2.VideoCapture(video_path)
        # Store the pixel sets for the first frame
    ret, frame = cap.read()
    shape = frame.shape
    lines = detect_lines(frame)
    prev_pixel_set = line_pixel_set(lines, shape)

    significant_movement_frames = []  # List to store frames where significant movement is detected
    frame_count = 0  # Frame counter

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Detect lines in the frame
        lines = detect_lines(frame)
        
        # Draw the detected lines on the frame
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Generate pixel sets for the current frame
        current_pixel_set = line_pixel_set(lines, shape)
        
        # Calculate the Jaccard distance between the pixel sets of the current and previous frames
        jaccard_distance = np.sum(np.bitwise_xor(prev_pixel_set, current_pixel_set)) / np.sum(np.bitwise_or(prev_pixel_set, current_pixel_set))
        
        if jaccard_distance > significant_movement_threshold:
            cv2.putText(frame, "Significant movement detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            significant_movement_frames.append(frame_count)  # Record the frame number
        
        # Update the previous pixel set
        prev_pixel_set = current_pixel_set
        
        # Display the resulting frame
        cv2.imshow('Line Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    print(significant_movement_frames)
    # Group significant movement frames into events
    events = event_filter(significant_movement_frames=significant_movement_frames, 
                          grouping_threshold=grouping_threshold, 
                          min_event_length=min_event_length)

    # Classify events as opening or closing alternately
    opening_closing_frames = [("Opening", event[len(event) >> 2]) if i % 2 == 0 else ("Closing", event[len(event) >> 1]) for i, event in enumerate(events)]

    # =========================================================================================================================================
    # if the len(opening_closing_frames) is larger than a threshold, use average algorithm to make the result smoother
    # =========================================================================================================================================

    if(len(opening_closing_frames) > 10):
        opening_closing_frames = detection(video_path, significant_movement_threshold * 1.1, grouping_threshold, min_event_length * 2)
    
    return opening_closing_frames


def detection_fast(video_path, significant_movement_threshold, grouping_threshold, min_event_length):

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    shape = frame.shape
    lines = detect_lines(frame)
    prev_pixel_set = line_pixel_set(lines, shape)

    significant_movement_frames = []  # List to store frames where significant movement is detected
    frame_count = 0  # Frame counter

    while True:

        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            break

        lines = detect_lines(frame)

        current_pixel_set = line_pixel_set(lines, shape)
        
        jaccard_distance = np.sum(np.bitwise_xor(prev_pixel_set, current_pixel_set)) / np.sum(np.bitwise_or(prev_pixel_set, current_pixel_set))
        
        if jaccard_distance > significant_movement_threshold:
            significant_movement_frames.append(frame_count)
        
        prev_pixel_set = current_pixel_set

    cap.release()
    events = event_filter(significant_movement_frames=significant_movement_frames, 
                          grouping_threshold=grouping_threshold, 
                          min_event_length=min_event_length)

    opening_closing_frames = [("Opening", event[len(event) >> 2]) if i % 2 == 0 else ("Closing", event[len(event) >> 1]) for i, event in enumerate(events)]

    if(len(opening_closing_frames) > 10):
        opening_closing_frames = detection_fast(video_path, significant_movement_threshold * 1.1, grouping_threshold, min_event_length * 2)
    

    return opening_closing_frames