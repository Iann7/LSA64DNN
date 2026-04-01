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
DATA_DIR = Path("data/raw")
POSE_DIR = Path("data/poses/real")
OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load your data
pose_path = next(POSE_DIR.glob("001_002_004.npy"), None)
video_path = next(DATA_DIR.glob("001_002_004.mp4"), None)

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
for frame_idx in tqdm(range(len(poses))):
    # Create a black canvas
    display_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # Get landmarks for current frame
    current_coords = poses[frame_idx]
    
    # Convert to MediaPipe Landmark List
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    skip_indices = {15,16,17,18,19,20,21,22,25, 26, 27, 28, 29, 30, 31, 32}
    print("==========================0")
    for i in range(len(current_coords)):
        if i >32:
            continue
       
        x, y, z = current_coords[i]
        if i==12:
            print(f"{x},{y},{z}")
        viz_x = (x*0.5) + 0.5
        viz_y = (y*0.5) + 0.5
        #if viz_x<0.51 and viz_x > 0.49 and viz_y<0.51 and viz_y > 0.49:
        #    print(f"index {i} is 0,0")
        # Rescale logic
        landmark_list.landmark.add(x=viz_x, y=viz_y, z=z)
    
    # Draw connections
    mp_drawing.draw_landmarks(
        display_frame, 
        landmark_list, 
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
    )
    
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