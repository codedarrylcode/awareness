
import pandas as pd
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
from umap import UMAP
from sklearn.cluster import KMeans

class TeamClassifier():
    def __init__(self, player_model, model_path = 'google/siglip-base-patch16-224', default_device = 'cuda', n_dim_reduced = 3, n_clusters = 2, strides = 30, start_frame = 0, end_frame = 1000):
        self.strides = 30
        self.frame_generator = None
        self.classes = None
        self.player_id = None
        self.start_frame = start_frame,
        self.end_frame = end_frame

        # Models
        self.player_model = player_model
        self.DEVICE = default_device if torch.cuda.is_available() else 'cpu'
        self.embeddings_model = SiglipVisionModel.from_pretrained(model_path).to(self.DEVICE)
        self.embeddings_processor = AutoProcessor.from_pretrained(model_path, use_fast = False)
        self.dim_reducer = UMAP(n_components = n_dim_reduced)
        self.hard_assigner = KMeans(n_clusters = n_clusters)

    def initialize_frames(self, source):
        self.frame_generator = sv.get_video_frames_generator(source_path = source, stride = self.strides, start = self.start_frame, end = self.end_frame)

    def crop_players(self, source, min_conf: float = 0.5, min_threshold: float = 0.5):
        # Get skipped frames of source
        self.initialize_frames(source)

        # Loop through each frame
        crops = []
        for i, frame in enumerate(self.frame_generator):
            result = self.player_model.predict(frame, conf = min_conf, verbose = False)
            detections = sv.Detections.from_ultralytics(result[0])
            detections = detections.with_nms(threshold = min_threshold, class_agnostic = True)

            # Get class_id for 'player'
            if i == 0:
                self.classes = (
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
                self.player_id = self.classes.loc[lambda x: x.class_name == 'player'].index[0]

            # Filter detections for players only
            player_detections = detections[detections.class_id == self.player_id]

            # Crop and store
            crops += [
                sv.crop_image(frame, xyxy)
                for xyxy in player_detections.xyxy
            ]

        return crops

    def transform_to_embeddings(self, crops, batch_size = 32):
        # Convert crops to batches of pillow images before embeddings
        batch_crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = chunked(batch_crops, batch_size)

        # Convert to embeddings
        data = []
        with torch.no_grad():
            for batch in batches:
                inputs = self.embeddings_processor(images = batch, return_tensors = 'pt').to(self.DEVICE)
                outputs = self.embeddings_model(**inputs)
                embeds = torch.mean(outputs.last_hidden_state, dim = 1).cpu().numpy()
                data.append(embeds)

        # Concatenate embeddings
        return np.concatenate(data)
    
    def fit(self, source, returnLabels = False):
        """Fit on frames from a source video"""
        # Crop players
        crops = self.crop_players(source)

        # Transform to embeddings
        embeddings = self.transform_to_embeddings(crops)

        # Reduce dimensionality
        projections = self.dim_reducer.fit_transform(embeddings)

        # Fit hard assigner
        self.hard_assigner.fit(projections)

    def predict(self, crops):
        """Predict for crops from a given frame"""
        embeddings = self.transform_to_embeddings(crops)
        projections = self.dim_reducer.transform(embeddings)
        labels = self.hard_assigner.predict(projections)

        return labels

def assign_goalies(players: sv.Detections, goalies: sv.Detections):
    # Get bottom edge of bbox for all players/goalies
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    goalie_xy = goalies.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    # Find average xy positions of each team
    team0_centroid = players_xy[players.class_id == 0].mean(axis = 0)
    team1_centroid = players_xy[players.class_id == 1].mean(axis = 0)

    # Find distances and assign goalies to closest team based on the average position in the frame
    goalie_team_id = []
    for xy in goalie_xy:
        dist0 = np.linalg.norm(xy - team0_centroid)
        dist1 = np.linalg.norm(xy - team1_centroid)

        team_id = 0 if dist0 < dist1 else 1
        goalie_team_id.append(team_id)

    # Store into goalies
    goalies.class_id = np.array(goalie_team_id)

    return goalies