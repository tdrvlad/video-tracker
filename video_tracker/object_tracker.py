import uuid
from typing import Optional
from typing import Union
from ultralytics import YOLO
import numpy as np
from typing import List
from itertools import chain

YOLO_MODEL = "./models/yolov3.pt"


class ObjectTracker:

    def __init__(self, model_path: str = YOLO_MODEL, interval: float = 0.5, batch_size: int = 2, target_class: Optional[Union[str, List[str]]] = None):
        """
        Initialize the DogTracker with the given model path and frame extraction interval.
        :param model_path: Path to the YOLO model file.
        :param interval: Interval in seconds between extracted frames.
        """
        self.model = YOLO(model_path)
        self.interval = interval
        self.batch_size = batch_size
        self.target_class = target_class
        self.trackers_dict = {}

        self.idx_to_classes_dict = {}
        self.classes_to_idx_dict = {}
        self.target_class = None

        if target_class is not None:
            self.process_target_class(target_class)

    def dummy_inference(self):
        dummy_input = np.random.random((244, 244, 3))
        dummy_result = self.model.track(source=dummy_input)
        return dummy_result

    def process_target_class(self, target_class: Union[str, List[str]]):
        dummy_result = self.dummy_inference()
        self.idx_to_classes_dict = dummy_result[0].names
        self.classes_to_idx_dict = {v: k for k, v in self.idx_to_classes_dict.items()}
        if isinstance(target_class, str):
            self.target_class = self.process_individual_target_class(target_class)
        elif isinstance(target_class, list):
            self.target_class = [self.process_individual_target_class(c) for c in target_class]

    def process_individual_target_class(self, target_class: str):
        if target_class not in self.classes_to_idx_dict:
            raise ValueError(f"Unrecognized class {target_class}. Must be one of {', '.join(self.classes_to_idx_dict.values())}")
        return self.classes_to_idx_dict[target_class]

    def batch_frames(self, frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        return [frames[i:i + self.batch_size] for i in range(0, len(frames), self.batch_size)]

    def track(self, frames: List[np.ndarray], inference_uuid: str = None):
        if inference_uuid is None:
            inference_uuid = str(uuid.uuid4())
            # self.model.predictor.trackers = []
            print(f"New inference stream added {inference_uuid}.")
        else:
            if inference_uuid not in self.trackers_dict:
                raise ValueError(f"UUID {inference_uuid} not recognized.")
            self.model.predictor.trackers = self.trackers_dict[inference_uuid]

        frames_batches = self.batch_frames(frames)
        results_batches = []
        for frames_batch in frames_batches:
            result = self.track_batch(
                frames_batch=frames_batch
            )
            processed_result = self.process_result(result)
            results_batches.append(processed_result)

        results = list(chain.from_iterable(results_batches))

        self.trackers_dict[inference_uuid] = self.model.predictor.trackers
        return results, inference_uuid

    def track_batch(self, frames_batch: List[np.ndarray]):
        results = self.model.track(
            source=frames_batch,
            persist=True,
            classes=self.target_class,
            batch=self.batch_size
        )
        return results

    @staticmethod
    def process_result(result_batch):
        processed_result_batch = []
        for frame_result in result_batch:
            result_dict = {}
            boxes = frame_result.boxes.xywh.cpu()
            if len(boxes):
                track_ids = frame_result.boxes.id.int().cpu().tolist()
                labels = frame_result.boxes.cls
                names = [frame_result.names[int(label)] for label in labels]

                for name, track_id, box in zip(names, track_ids, boxes):
                    result_dict[f"#{track_id}-{name}"] = box

            processed_result_batch.append(result_dict)

        return processed_result_batch
