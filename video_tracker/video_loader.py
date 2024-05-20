import cv2
import numpy as np
from typing import List, Tuple


class VideoLoader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) / self.frame_rate

    def get_video_length(self) -> float:
        """
        Get the length of the loaded video in seconds.
        :return: Length of the video in seconds.
        """
        if self.video_length is None:
            raise ValueError("Video not loaded.")
        return self.video_length

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """
        Get the width and height of the frames of the loaded video.
        :return: A tuple containing (width, height) of the frames.
        """
        if self.frame_width is None or self.frame_height is None:
            raise ValueError("Video not loaded.")
        return self.frame_width, self.frame_height

    def extract_frames_and_timestamps(self, interval: float) -> List[Tuple[np.ndarray, float]]:
        """
        Extract frames from the video at the given rate.
        :param interval: Interval in seconds between selected frames.
        :return: List of tuples containing the frame and the timestamp.
        """
        if self.cap is None:
            raise ValueError("Video not loaded.")

        frames = []
        current_time = 0.0

        while self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append((frame, current_time))
            current_time += interval

        self.cap.release()
        return frames

    def extract_frames(self, interval: float) -> List[np.ndarray]:
        frames_and_timestamps = self.extract_frames_and_timestamps(interval=interval)
        frames = [ft[0] for ft in frames_and_timestamps]
        return frames


def test():
    vp = VideoLoader("./data/sample-1.mp4")
    print(vp.get_video_length())
    print(vp.get_frame_dimensions())
    frames = vp.extract_frames(0.5)
    for frame, timestamp in frames:
        print(f"Timestamp: {timestamp}, Frame shape: {frame.shape}")


if __name__ == "__main__":
    test()

