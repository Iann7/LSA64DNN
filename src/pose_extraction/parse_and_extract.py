from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm 
import cv2  
import numpy as np 
import pandas as pd 

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.pose_extraction.shift_landmarks import process_sequence_pose
# Paths
DATA_DIR = Path("data/raw")
POSE_DIR = Path("data/poses/shifted_and_blurred")
METADATA_DIR = Path("data/metadata")
POSE_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Video Files
video_files = list(DATA_DIR.glob("*.mp4")) 

# Sign Names
SIGN_NAMES = {
    1: "Opaque", 2: "Red", 3: "Green", 4: "Yellow", 5: "Bright",
    6: "Light-blue", 7: "Colors", 8: "Pink", 9: "Women", 10: "Enemy",
    11: "Son", 12: "Man", 13: "Away", 14: "Drawer", 15: "Born",
    16: "Learn", 17: "Call", 18: "Skimmer", 19: "Bitter", 20: "Sweet milk",
    21: "Milk", 22: "Water", 23: "Food", 24: "Argentina", 25: "Uruguay",
    26: "Country", 27: "Last name", 28: "Where", 29: "Mock", 30: "Birthday",
    31: "Breakfast", 32: "Photo", 33: "Hungry", 34: "Map", 35: "Coin",
    36: "Music", 37: "Ship", 38: "None", 39: "Name", 40: "Patience",
    41: "Perfume", 42: "Deaf", 43: "Trap", 44: "Rice", 45: "Barbecue",
    46: "Candy", 47: "Chewing-gum", 48: "Spaghetti", 49: "Yogurt",
    50: "Accept", 51: "Thanks", 52: "Shut down", 53: "Appear", 54: "To land",
    55: "Catch", 56: "Help", 57: "Dance", 58: "Bathe", 59: "Buy",
    60: "Copy", 61: "Run", 62: "Realize", 63: "Give", 64: "Find"
}   

def parse_and_extract():
    num_workers = 4
    print(f"Starting parallel processing with {num_workers} cores!")
    
    all_labels = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_video = {executor.submit(process_single_video, video_path): video_path for video_path in video_files}
        for future in tqdm(as_completed(future_to_video), total=len(video_files), desc="Processing Videos"):
            result = future.result() 
            if result.get('success'):
                all_labels.append(result)

    if all_labels:
        data_frame = pd.DataFrame(all_labels)
        data_frame.drop(columns=['success'], inplace=True)
        data_frame.to_csv(METADATA_DIR / 'real_labels.csv', index=False)
        print(f"\nSaved labels to {METADATA_DIR / 'real_labels.csv'}")  
        print(f"Total: {len(data_frame)} videos processed")

def process_single_video(video_path):
    try:
        pose, hands = init_mediapipe()
        
        stem = video_path.stem  
        parts = stem.split('_')  
        sign_id = int(parts[0])
        
        video_landmarks = parse_video(video_path, pose, hands)
        pose.close()
        hands.close()

        if len(video_landmarks) > 0:
            poses_array = np.array(video_landmarks)
            output_path = POSE_DIR / f"{stem}.npy"
            np.save(output_path, poses_array)
            
            return {
                'filename': f"{stem}.npy",
                'sign_id': sign_id,
                'subject': int(parts[1]),
                'repetition': int(parts[2]),
                'sign_name': SIGN_NAMES.get(sign_id, "Unknown"),
                'num_frames': len(video_landmarks),
                'success': True
            }
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
    
    return {'success': False}


def init_mediapipe():
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="models/pose_landmarker_lite.task"),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.65,
        min_pose_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )
    pose = vision.PoseLandmarker.create_from_options(pose_options)
    
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )
    hands = vision.HandLandmarker.create_from_options(hand_options)
    
    return pose, hands

def parse_video(video_path, pose, hands):
    cap = cv2.VideoCapture(str(video_path))
    landmarks = [] 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        landmark = extract_landmark_from_frame(frame, pose, hands)
        landmarks.append(landmark)
    cap.release()
    return landmarks

def extract_landmark_from_frame(frame, pose, hands):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    pose_result = get_pose_coords(pose.detect(mp_image), 33)
    hand_result = extract_hand_coords(hands.detect(mp_image))
    
    return np.concatenate([pose_result, hand_result])

def get_pose_coords(res, num_landmarks):
    if res and res.pose_landmarks and len(res.pose_landmarks) > 0:
        return [val for lm in res.pose_landmarks[0] for val in [lm.x, lm.y, lm.z]]
    else:
        return [0.0] * (num_landmarks * 3) 
    
def extract_hand_coords(hand_result):
    output_coords = [0.0] * (42 * 3) 
    
    if not hand_result or not hand_result.hand_landmarks:
        return output_coords
        
    for idx, handedness in enumerate(hand_result.handedness):
        label = handedness[0].category_name # "Left" or "Right"
        landmarks = hand_result.hand_landmarks[idx]
        
        temp_coords = []
        for lm in landmarks:
            temp_coords.extend([lm.x, lm.y, lm.z])
            
        if label == "Right":
            output_coords[0:63] = temp_coords  
        else:
            output_coords[63:126] = temp_coords 
            
    return output_coords

if __name__ == "__main__":
    parse_and_extract()