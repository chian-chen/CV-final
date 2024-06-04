import cv2
import numpy as np
from vidstab import VidStab
# from vidstab import VidStab

from utils import detect_lines, line_pixel_set, event_filter


def stabilize_video(input_path, output_path):
    stabilizer = VidStab()
    stabilizer.stabilize(input_path=input_path, output_path=output_path)


def average_frames(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frames_buffer = []
    
    for _ in range(3):
        ret, frame = cap.read()
        if not ret:
            break
        frames_buffer.append(frame)

    frame_idx = 0

    while True:
        if len(frames_buffer) < 3:
            break

        avg_frame = np.mean(frames_buffer, axis=0).astype(np.uint8)
        out.write(avg_frame)

        frames_buffer.pop(0)

        ret, frame = cap.read()
        if not ret:
            break
        frames_buffer.append(frame)

        frame_idx += 1

    cap.release()
    out.release()


if __name__ == '__main__':
    # stabilize_video('./samples/02.mp4', '02_stable.mp4')
    average_frames('./samples/02.mp4', '02_stable.mp4')