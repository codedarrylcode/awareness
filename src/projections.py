import numpy as np
import cv2
import supervision as sv

class Projector:
    def __init__(self, min_conf = 0.5):
        self.conf = 0.5
        self.key_points = None
        self.anchors_filter = None
        self.anchors = None

    def detect_keypoints(self, frame, detector):
        result = detector.predict(frame, verbose = False)[0]
        self.key_points = sv.KeyPoints.from_ultralytics(result)

        return self.key_points

    def detect_anchors(self):
        # Define filter for anchors
        self.anchors_filter = self.key_points.confidence[0] > self.conf

        # Filter
        self.anchors = self.key_points.xy[0][self.anchors_filter]

        return self.anchors
    
    def fetch_pitch_points(self, config):
        return np.array(config.vertices)[self.anchors_filter]
    
    def project_objects_to_2D(self, detections, transformer):
        projected = {}
        for class_name, det in detections.items():
            xy = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            xy = transformer.transform_points(points = xy)
            projected[class_name] = xy

        return projected
    
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        Find the relevant positions of the target anchor points on a 2D pitch's reference points
        """
        # Ensure float type prior to computations
        source = source.astype(np.float32) # Anchor points
        target = target.astype(np.float32) # Relevant-to-anchor 2D pitch points

        # Calculate homography matrix - matrix that matches source to target
        self.m, _ = cv2.findHomography(source, target)

        # Check for Nonetypes
        if self.m is None:
            return ValueError("Homography matrix could not be calculated")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Given all keypoints of a 2D pitch, use the initialized Homography matrix to project anchor points into the pitch.

        i.e. `points` is an argument that contains 2D pitch keypoints, while `self.m` contains the Homography matrix
        """
        # Shape points dimensionality to fit homography matrix's shape
        points = points.reshape(-1, 1, 2).astype(np.float32)

        # Use homography matrix to transform perspective
        points = cv2.perspectiveTransform(points, self.m)

        return points.reshape(-1, 2).astype(np.float32)
    
def get_positions_by_teams(projections, detections, withGoalies = True):
    if withGoalies:
        team1_xy = np.concatenate([
            projections['player'][detections['player'].class_id == 0],
            projections['goalkeeper'][detections['goalkeeper'].class_id == 0]
        ])

        team2_xy = np.concatenate([
            projections['player'][detections['player'].class_id == 1],
            projections['goalkeeper'][detections['goalkeeper'].class_id == 1]
        ])

    else:
        team1_xy = projections['player'][detections['player'].class_id == 0]
        team2_xy = projections['player'][detections['player'].class_id == 1]

    return (team1_xy, team2_xy)