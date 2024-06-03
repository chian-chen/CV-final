import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from utils import detect_lines, line_pixel_set

# def arg_parse():
#     parser = argparse.ArgumentParser(description='CV_final')
    
#     parser.add_argument('--video_path', default='./01.mp4', type=int, help='path to test video')
#     args = parser.parse_args()
#     return args

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
        idx = len(event) >> 1
        if i % 2 == 0:
            idx = idx >> 1
            opening_closing_frames.append(("opening", event[idx]))
        else:
            opening_closing_frames.append(("closing", event[idx]))

    # Print the events in alternating order
    for event_type, frame in opening_closing_frames:
        print(f"Door {event_type} at frame {frame}")
    if len(opening_closing_frames) % 2 != 0 and opening_closing_frames[-1][0] == "opening":
        print(f"Door remains open at frame {opening_closing_frames[-1][1]}")

    return opening_closing_frames

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



if __name__ == '__main__':
    path = './tests/'
    videos = ['01.mp4', '03.mp4', '05.mp4', '07.mp4', '09.mp4']
    # path = './samples'
    # videos = ['01.mp4', '02.mp4', '03.mp4']

    # ====================================================================================
    # SETTING
    significant_movement_threshold = 0.6  # Threshold for significant movement
    grouping_threshold = 10  # Number of frames to consider movements as part of the same event
    min_event_length = 3  # Minimum number of frames for a valid event
    # ====================================================================================

    results = []

    for video in videos:
        video_path = os.path.join(path, video)
        opening_closing_frames = detection(video_path=video_path, 
                  significant_movement_threshold=significant_movement_threshold,
                  grouping_threshold=grouping_threshold,
                  min_event_length=min_event_length)
        results.append(opening_closing_frames)

    print(f'Final Score in Test Data: {evaluation(results=results)}')

        