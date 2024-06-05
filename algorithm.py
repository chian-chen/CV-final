import os
import json
import argparse

from utils import detection_fast

def scan_videos(directory, args):
    """Scan the specified directory for MP4 files and generate JSON annotations."""
    video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    videos_info = []

    for video_file in video_files:
        opening_closing_frames = detection_fast(
                  video_path=os.path.join(directory, video_file), 
                  significant_movement_threshold=args.significant_movement_threshold,
                  grouping_threshold=args.grouping_threshold,
                  min_event_length=args.min_event_length
        )
        states = [{"state_id": i + 1,
                   "description": info[0],\
                    "guessed_frame": info[1]} for i, info in enumerate(opening_closing_frames)]
        
        videos_info.append({
            "video_filename": video_file,
            "annotations": [
                {
                    "object": "Door",
                    "states": states
                }
            ]
        })

    return videos_info

def generate_json(output_filename, videos_info):
    """Generate a JSON file with the provided video information."""
    with open(output_filename, 'w') as file:
        json.dump({"videos": videos_info}, file, indent=4)

def arg_parse():
    parser = argparse.ArgumentParser(description='CV_final')
    
    parser.add_argument('--significant_movement_threshold', default=0.6, type=float, help='Threshold for significant movement')
    parser.add_argument('--grouping_threshold', default=10, type=int, help='Number of frames to consider movements as part of the same event')
    parser.add_argument('--min_event_length', default=5, type=int, help='Minimum number of frames for a valid event')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    directory = "./tests"  # Specify the directory to scan
    output_filename = "algorithm_output.json"  # Output JSON file name
    args = arg_parse()

    videos_info = scan_videos(directory, args)
    generate_json(output_filename, videos_info)
    print(f"Generated JSON file '{output_filename}' with video annotations.")
