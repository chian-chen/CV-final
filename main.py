import argparse
import os

from utils import detection



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

        