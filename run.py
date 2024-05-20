from itertools import chain

import numpy as np

from video_tracker.object_tracker import ObjectTracker
from video_tracker.video_loader import VideoLoader
from video_tracker.visualize import create_video_with_detections


def run(video_path: str = "./data/sidewalk.mp4", sampling_interval: float = 0.1):
    """
    Demo the ObjectTracker on a video.
    :param video_path: Path to the video
    :param sampling_interval: Interval between frames (seconds)
    """
    tracker = ObjectTracker()
    fps = 1 / sampling_interval
    video = VideoLoader(video_path)

    frames = video.extract_frames(sampling_interval)

    k = 2
    """
    Note: In a practical application, the video stream might be processed in parallel with other video streams.
    Therefore it is necessary to keep the tracking separated between streams.
    
    We simulate this by running the inferences in groups of 2 frames at a time, with different stream being intertwined.
    The goal is to see that the final results maintain consistency between object ids.
    """

    frames_chunks = [frames[i:i + k] for i in range(0, len(frames), k)]
    inference_uuid = None
    """
    Inference uuid allows us to keep track of the inferences streams
    """

    results = []
    for frames_chunk in frames_chunks:
        result, inference_uuid = tracker.track(frames_chunk, inference_uuid=inference_uuid)
        results.append(result)

        """ We simulate a different video stream with a random input. """
        dummy_result, other_inference_uuid = tracker.track([np.random.random((224, 224, 3))])

    results = list(chain.from_iterable(results))
    create_video_with_detections(
        frames=frames,
        detections=results,
        output_path='./data/result.mp4',
        fps=int(fps)
    )


if __name__ == '__main__':
    run()
