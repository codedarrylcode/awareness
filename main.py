# Default imports
import os
import matplotlib.pyplot as plt
import supervision as sv
import cv2
import ffmpeg
from ultralytics import YOLO

# Relative imports
from src.team_assigner import TeamClassifier, assign_goalies
from src.detectors import Detector
from src.annotations import SoccerPitchConfiguration, draw_radar, draw_voronoi, annotate_players, annotate_ball, annotate_with_radar
from src.projections import ViewTransformer, Projector, get_positions_by_teams

# Suppress supervision warnings
import warnings
warnings.simplefilter(action = 'ignore')

# Define specifications
PROJECT_DIR = os.getcwd()
SOURCE_VIDEO_PATH = f'{PROJECT_DIR}/data/train/08fd33_4.mp4'
PLAYER_DETECTION_MODEL_PATH = f'{PROJECT_DIR}/weights/football-player-detection-v9.pt'
BALL_DETECTION_MODEL_PATH = f'{PROJECT_DIR}/weights/football-ball-detection-v2.pt'
PITCH_DETECTION_MODEL_PATH = f'{PROJECT_DIR}/weights/football-pitch-detection-v9.pt'
CODEC = 'XVID'
START_FRAME, END_FRAME = (0, 30) # Read from start till N frames

def main():
    # Initialize frame/video helpers, tracker, detection model(s), soccer pitch configs
    frame_gen = sv.get_video_frames_generator(source_path = SOURCE_VIDEO_PATH, start = START_FRAME, end = END_FRAME)
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH) # Stores vid info (weight, height, fps, frames)
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    tracker = sv.ByteTrack()
    player_detector = YOLO(PLAYER_DETECTION_MODEL_PATH)
    ball_detector = YOLO(BALL_DETECTION_MODEL_PATH)
    pitch_detector = YOLO(PITCH_DETECTION_MODEL_PATH)
    CONFIG = SoccerPitchConfiguration()

    # Build Team Classifier model (10 mins)
    clf = TeamClassifier(player_model = player_detector)
    clf.fit(SOURCE_VIDEO_PATH)

    # Run for each frame
    outputs = []
    for frame in frame_gen:
        # Detect objects
        detector = Detector()
        detections = detector.detect(frame, player_detector)
        detections = tracker.update_with_detections(detections)
        detections = detector.split_detections(detections)
        detections = detector.detect_ball(frame, detections, ball_detector) # fix ball detections

        # Resolve team assignments
        player_crops = [sv.crop_image(frame, xyxy) for xyxy in detections['player'].xyxy]
        labels = clf.predict(player_crops)
        detections['player'].class_id = labels
        goaliesDetected = 'goalkeeper' in detections.keys()
        if goaliesDetected:
            detections['goalkeeper'] = assign_goalies(detections['player'], detections['goalkeeper']) # Goalie's shortest dist to each team's avg position

        # Detect key points
        projector = Projector()
        key_points = projector.detect_keypoints(frame, pitch_detector)
        anchors = projector.detect_anchors()

        # Project to 2D pitch
        pitch_points = projector.fetch_pitch_points(CONFIG)
        transformer = ViewTransformer(source = anchors, target = pitch_points)
        projections = projector.project_objects_to_2D(detections, transformer)

        # Get positions of players from each team, with/without goalies
        team1_players_xy, team2_players_xy = get_positions_by_teams(projections, detections, withGoalies = False) # Without goalies

        if goaliesDetected:
            team1_xy, team2_xy = get_positions_by_teams(projections, detections)
        else:
            team1_xy, team2_xy = team1_players_xy, team2_players_xy

        # Annotate
        # Draw radar
        radar = draw_radar(
            config = CONFIG,
            projections = projections,
            detections = detections,
            team_1_xy = team1_players_xy,
            team_2_xy = team2_players_xy
        )

        # Draw voronoi
        voronoi = draw_voronoi(
            config = CONFIG,
            team_1_xy = team1_xy,
            team_2_xy = team2_xy,
        )

        # Draw voronoi with radar
        voronoi_radar = draw_voronoi(
            config = CONFIG,
            team_1_xy = team1_xy,
            team_2_xy = team2_xy,
            projections = projections,
            detections = detections,
        )

        # Draw voronoi with blending
        voronoi_blended = draw_voronoi(
            config = CONFIG,
            team_1_xy = team1_xy,
            team_2_xy = team2_xy,
            projections = projections,
            detections = detections,
            isBlended = True
        )

        # Annotate objects on frame
        annotated = frame.copy()
        annotated = annotate_players(annotated, detections)

        if 'ball' in detections.keys():
            annotated = annotate_ball(annotated, detections)

        if 'goalkeeper' in detections.keys():
            annotated = annotate_players(annotated, detections, objects = 'goalkeeper')

        # Embed a radar into an annotated frame
        combi = annotated.copy()
        small_radar = voronoi_blended.copy()
        combi = annotate_with_radar(combi, small_radar, alpha = 1)

        # Store
        annos = {
            'radar' : radar,
            'voronoi' : voronoi,
            'voronoi_with_radar' : voronoi_radar,
            'voronoi_blended' : voronoi_blended,
            'tracking' : annotated,
            'tracking_with_radar' : combi,
        }
        outputs.append(annos)

    # Write frames to video output
    output_types = annos.keys()

    for output_type in output_types:
        # Initialize video writer
        output_path = f'{PROJECT_DIR}/videos/output/{output_type}.avi'
        mp4_path = f'{PROJECT_DIR}/videos/output/{output_type}.mp4'
        if 'tracking' in output_type:
            video_writer = cv2.VideoWriter(output_path, fourcc, video_info.fps, video_info.resolution_wh)
        else:
            video_writer = cv2.VideoWriter(output_path, fourcc, video_info.fps, (outputs[0][output_type].shape[1], outputs[0][output_type].shape[0]))

        # Write output
        for output in outputs:
            video_writer.write(output[output_type])

        # Close writer
        video_writer.release()

        # Run video compresion
        try:
            (
                ffmpeg
                .input(output_path)
                .output(mp4_path)
                .run(overwrite_output = True, capture_stdout = True, capture_stderr = True)
            )
        except ffmpeg.Error as e:
            print('stderr:', e.stderr.decode() if e.stderr else None)

if __name__ == '__main__':
    main()