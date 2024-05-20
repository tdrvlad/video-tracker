import random

import cv2


def get_color(label):
    """
    Generate a random color for each label.

    Parameters:
        label (str): The label for which the color is to be generated.

    Returns:
        tuple: A tuple representing the color in BGR format.
    """
    random.seed(hash(label) % 2 ** 32)  # Seed with the hash of the label
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def draw_boxes(frame, detections):
    """
    Draw bounding boxes on the frame based on the detections.

    Parameters:
        frame (numpy array): The image frame.
        detections (dict): A dictionary with keys as object labels and values as bounding box coordinates.

    Returns:
        frame (numpy array): The frame with bounding boxes drawn.
    """
    for label, bbox in detections.items():
        bbox = bbox.numpy()  # Convert tensor to numpy array if necessary
        cx, cy, w, h = bbox
        x_min = int(cx - w / 2)
        y_min = int(cy - h / 2)
        x_max = int(cx + w / 2)
        y_max = int(cy + h / 2)
        color = get_color(label)
        # Draw rectangle
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        # Put label text
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def create_video_with_detections(frames, detections, output_path, fps=30):
    """
    Create a video with the detection results overlaid on the frames.

    Parameters:
        frames (list of numpy arrays): List of frames (images) as numpy arrays.
        detections (list of dicts): List of dictionaries containing detections for each frame.
        output_path (str): Path to save the output video.
        fps (int): Frames per second for the output video.
    """
    # Get frame dimensions
    height, width, _ = frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, frame in enumerate(frames):
        # Get detections for the current frame
        frame_detections = detections[i]
        # Draw bounding boxes on the frame
        frame_with_boxes = draw_boxes(frame.copy(), frame_detections)
        # Write the frame into the video
        out.write(frame_with_boxes)

    # Release everything if job is finished
    out.release()
    print(f"Video saved at {output_path}")