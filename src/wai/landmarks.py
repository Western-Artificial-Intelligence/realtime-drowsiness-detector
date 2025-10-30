import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
from typing import Sequence
from dataclasses import dataclass

@dataclass
class LandmarkFrame:
    landmarks: Sequence[NormalizedLandmark]
    frame_width: int
    frame_height: int
    annotated_frame: np.ndarray | None = None

class FaceMeshDetector:
    def __init__(self, max_faces=1, det_conf=0.5, track_conf=0.5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_faces,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
            refine_landmarks=True,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
    
    def process(self, frame_bgr) -> LandmarkFrame | None:
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
        frame_height, frame_width = frame_bgr.shape[:2]

        annotated_frame = frame_bgr.copy()
        frame_height, frame_width = frame_bgr.shape[:2]

        # for lm in face_landmarks.landmark:
        #     x = int(lm.x*frame_width)
        #     y = int(lm.y*frame_height)
        #     cv2.circle(annotated_frame, (x, y), radius=1, color=(0,255,0), thickness=-1)
        
        self.mp_drawing.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        self.mp_drawing.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )

        self.mp_drawing.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )

        return LandmarkFrame(
            landmarks=landmarks,
            frame_width=frame_width,
            frame_height=frame_height,
            annotated_frame=annotated_frame
        )