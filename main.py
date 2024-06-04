import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from utils import detect_lines, line_pixel_set, evaluation, event_filter


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
        # if lines is not None:
        #     for line in lines:
        #         for x1, y1, x2, y2 in line:
        #             cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Generate pixel sets for the current frame
        current_pixel_set = line_pixel_set(lines, shape)
        
        # Calculate the Jaccard distance between the pixel sets of the current and previous frames
        jaccard_distance = np.sum(np.bitwise_xor(prev_pixel_set, current_pixel_set)) / np.sum(np.bitwise_or(prev_pixel_set, current_pixel_set))
        
        if jaccard_distance > significant_movement_threshold:
            # cv2.putText(frame, "Significant movement detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            significant_movement_frames.append(frame_count)  # Record the frame number
        
        # Update the previous pixel set
        prev_pixel_set = current_pixel_set
        
        # Display the resulting frame
        # cv2.imshow('Line Detection', frame)

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
    opening_closing_frames = [("opening", event[len(event) >> 2]) if i % 2 == 0 else ("closing", event[len(event) >> 1]) for i, event in enumerate(events)]

    # =========================================================================================================================================
    # if the len(opening_closing_frames) is larger than a threshold, use average algorithm to make the result smoother
    # =========================================================================================================================================

    if(len(opening_closing_frames) > 10):
        opening_closing_frames = detection(video_path, significant_movement_threshold * 1.1, grouping_threshold, min_event_length * 2)
    
    return opening_closing_frames


def arg_parse():
    parser = argparse.ArgumentParser(description='CV_final')
    
    parser.add_argument('--significant_movement_threshold', default=0.6, type=float, help='Threshold for significant movement')
    parser.add_argument('--grouping_threshold', default=10, type=int, help='Number of frames to consider movements as part of the same event')
    parser.add_argument('--min_event_length', default=5, type=int, help='Minimum number of frames for a valid event')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # path = './tests/'
    # videos = ['01.mp4', '03.mp4', '05.mp4', '07.mp4', '09.mp4']
    path = './samples'
    videos = ['01.mp4', '02.mp4', '03.mp4']

    args = arg_parse()

    results = []
    for video in videos:
        video_path = os.path.join(path, video)
        opening_closing_frames = detection(
                  video_path=video_path, 
                  significant_movement_threshold=args.significant_movement_threshold,
                  grouping_threshold=args.grouping_threshold,
                  min_event_length=args.min_event_length
        )

        for event_type, frame in opening_closing_frames:
            print(f"Door {event_type} at frame {frame}")
        if len(opening_closing_frames) % 2 != 0 and opening_closing_frames[-1][0] == "opening":
            print(f"Door remains open at frame {opening_closing_frames[-1][1]}")
    
        results.append(opening_closing_frames)
    # print(f'Final Score in Test Data: {evaluation(results=results)}')

        