import supervision as sv
import pandas as pd
import numpy as np
from collections import deque

class BallTracker:
    """
    A class used to track a soccer ball's position across video frames.

    The BallTracker class maintains a buffer of recent ball positions and uses this
    buffer to predict the ball's position in the current frame by selecting the
    detection closest to the average position (centroid) of the recent positions.

    Attributes:
        buffer (collections.deque): A deque buffer to store recent ball positions.
    """
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the buffer with new detections and returns the detection closest to the
        centroid of recent positions.

        Args:
            detections (sv.Detections): The current frame's ball detections.

        Returns:
            sv.Detections: The detection closest to the centroid of recent positions.
            If there are no detections, returns the input detections.
        """
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]

class Detector:
    def __init__(self, min_conf = 0.5, nms_threshold = 0.5):
        self.conf = min_conf
        self.threshold = nms_threshold
        self.class_df = None

    def detect(self, frame, detector):
        # Run detections
        results = detector.predict(frame, conf = self.conf, verbose = False)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = detections.with_nms(threshold = self.threshold)

        return detections
    
    def store_classes(self, detections):
        self.class_df = (
            pd.DataFrame(
                np.array(
                    [
                        detections.class_id,
                        detections.data['class_name']
                    ]
                ).T,
                columns = ['class_id', 'class_name']
            )
            .drop_duplicates()
            .reset_index(drop = True)
            .astype({'class_id': int})
            .set_index('class_id')
            .sort_index()
        )

    def split_detections(self, detections):
        if self.class_df is None:
            self.store_classes(detections)
            
        return {class_name: detections[detections.class_id == class_id] for class_id, class_name in self.class_df.class_name.to_dict().items()}

    def detect_ball(self, frame, detections, ball_detector, imgsz = 640, slice_wh = (640, 640), buffer_size = 20, nms_threshold = 0.1, padding = 10):
        # Use a callback function to feed into slicer
        def callback(image_slice: np.ndarray) -> sv.Detections:
            result = ball_detector(image_slice, imgsz = imgsz, verbose = False)[0]
            return sv.Detections.from_ultralytics(result)
        
        # Initialize slicer tool
        slicer = sv.InferenceSlicer(
            callback = callback,
            overlap_filter = sv.OverlapFilter.NONE, # Deals with overlapping frames, improves detection accuracy
            slice_wh = slice_wh,
        )

        # Get detections
        ball_detections = slicer(frame).with_nms(threshold=nms_threshold) # with non-max suppression
        if ball_detections.xyxy.shape[0]:
            detections['ball'] = ball_detections

            # Update with BallTacker
            ball_tracker = BallTracker(buffer_size = buffer_size)
            detections['ball'] = ball_tracker.update(detections['ball'])

            # Pad the ball detection (slightly)
            detections['ball'].xyxy = sv.pad_boxes(xyxy = detections['ball'].xyxy, px = padding)

        return detections
