import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from pathlib import Path
from tqdm import tqdm

# Setup drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Define paths
DATA_DIR = Path("data/real")
POSE_DIR = Path("data/poses/real")
OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load your data
pose_path = next(POSE_DIR.glob("001_004_002.npy"), None)
video_path = next(DATA_DIR.glob("001_004_002.mp4"), None)

if pose_path is None:
    raise FileNotFoundError("No pose .npy file found in data/poses/real/")

poses = np.load(pose_path)
poses = poses.reshape(-1, 75, 3)

# Setup video writer
frame_height, frame_width = 600, 600
fps = 30
output_path = OUTPUT_DIR / "skeleton_visualization.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

# Visualization Loop
print(f"Processing {len(poses)} frames...")
def to_mp_list(pose_coords):
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    print("==========================")
    for i in range(len(pose_coords)):
        x, y, z = pose_coords[i]
        viz_x = (x*0.7) 
        viz_y = (y*0.7) 
        landmark_list.landmark.add(x=viz_x, y=viz_y, z=z)
    return landmark_list

for frame_idx in range(len(poses)):
    # Create a black canvas
    display_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # Get landmarks for current frame
    current_coords = poses[frame_idx]
    pose_coords = current_coords[0:33]
    hand_right_coords = current_coords[33:54]
    hand_left_coords = current_coords[54:75]
    # Convert to MediaPipe Landmark List
    pose_landmark_list = to_mp_list(pose_coords)
    hand_left_landmark_list = to_mp_list(hand_left_coords)
    hand_right_landmark_list = to_mp_list(hand_right_coords)
    # Draw connections
    mp_drawing.draw_landmarks(
        display_frame, 
        pose_landmark_list, 
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
    )
    # MANO DERECHA
    mp_drawing.draw_landmarks(display_frame, to_mp_list(hand_right_coords), mp_holistic.HAND_CONNECTIONS)

    # MANO IZQUIERDA
    mp_drawing.draw_landmarks(display_frame, to_mp_list(hand_left_coords), mp_holistic.HAND_CONNECTIONS)
    # Write frame to video  
    out.write(display_frame)
    
    # Show preview (optional - comment out if too slow)
    cv2.imshow('Skeleton Visualization', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
out.release()
cv2.destroyAllWindows()
print(f"\n✓ Video saved to: {output_path}")