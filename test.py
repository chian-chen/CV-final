import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def detect_lines(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection
    edged = cv2.Canny(blurred, 50, 100)
    # edged = cv2.Canny(blurred, 50, 100)
    # Detect lines using the Hough Line Transform
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)
    return lines, edged

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

def arg_parse():
    parser = argparse.ArgumentParser(description='CV_final')
    
    parser.add_argument('--video_path', default='./tests/09.mp4', type=str, help='path to test video')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = arg_parse()
    # Initialize the video capture
    # cap = cv2.VideoCapture('./tests/09.mp4')
    cap = cv2.VideoCapture(args.video_path)

    # Store the pixel sets for the first frame
    ret, frame = cap.read()
    lines, edged = detect_lines(frame)
    shape = edged.shape
    prev_pixel_set = line_pixel_set(lines, shape)

    significant_movement_threshold = 0.6 * 1.1  # Threshold for significant movement
    grouping_threshold = 10 # Number of frames to consider movements as part of the same event
    min_event_length = 5 * 2  # Minimum number of frames for a valid event

    significant_movement_frames = []  # List to store frames where significant movement is detected
    frame_count = 0  # Frame counter

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Detect lines in the frame
        lines, edged = detect_lines(frame)
        
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

    # Classify events as opening or closing alternately
    opening_closing_frames = []

    for i, event in enumerate(events):
        if i % 2 == 0:
            opening_closing_frames.append(("opening", event[0]))
        else:
            opening_closing_frames.append(("closing", event[0]))

    # Print the events in alternating order
    for event_type, frame in opening_closing_frames:
        print(f"Door {event_type} at frame {frame}")
    if len(opening_closing_frames) % 2 != 0 and opening_closing_frames[-1][0] == "opening":
        print(f"Door remains open at frame {opening_closing_frames[-1][1]}")

    # # Plot the results
    # plt.figure(figsize=(10, 6))
    # plt.plot(significant_movement_frames, [1] * len(significant_movement_frames), 'ro', markersize=5)
    # plt.title('Significant Movement Detection')
    # plt.xlabel('Frame Number')
    # plt.ylabel('Detection')
    # plt.yticks([0, 1], ['', 'Detected'])
    # plt.show()
