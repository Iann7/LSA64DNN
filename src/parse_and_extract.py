from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from tqdm import tqdm 
import os 
import mediapipe as mp
import cv2  
import numpy as np 
import re 
import pandas as pd 

# Paths
DATA_DIR = Path("data/raw")
POSE_DIR = Path("data/poses/real")
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
    num_workers = 2
    print(f"Starting parallel processing with {num_workers} cores!")
    
    all_labels = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_video = {executor.submit(process_single_video, video_path): video_path for video_path in video_files}
        for future in tqdm(as_completed(future_to_video),total=len(video_files),desc="Processing Videos"):
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
        # 1. Initialize inside the worker process
        holistic_model = init_mediapipe()
        
        stem = video_path.stem  
        parts = stem.split('_')  
        sign_id = int(parts[0])
        
        # 2. Pass the model instance to the parser
        video_landmarks = parse_video(video_path, holistic_model)
        
        # Clean up the model to free memory in the subprocess
        holistic_model.close()

        if len(video_landmarks) > 0:
            poses_array = np.array(video_landmarks)
            output_path = POSE_DIR / f"{stem}.npy"
            np.save(output_path, poses_array)
            
            return {
                'filename': f"{stem}.npy",
                'sign_id': sign_id,
                'subject': int(parts[1]),
                'repetition': int(parts[2]),
                'sign_name': SIGN_NAMES[sign_id],
                'num_frames': len(video_landmarks),
                'success': True
            }
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
    
    return {'success': False}

def init_mediapipe():
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    refine_face_landmarks=False)
    return holistic

def parse_video(video_path, holistic):
    cap = cv2.VideoCapture(str(video_path))
    landmarks = [] 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 3. Pass holistic through here
        landmark = extract_landmark_from_frame(frame, holistic)
        landmarks.append(landmark)
    cap.release()
    return landmarks

def extract_landmark_from_frame(frame,holistic):

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(rgb_frame)
    pose = get_coords(result.pose_landmarks, 33)
    lh = get_coords(result.left_hand_landmarks, 21)
    rh = get_coords(result.right_hand_landmarks, 21)
    #Faces are left out due to LSA64 not using facial gestures at all
    return normalize_relative(pose, lh, rh) 

def get_coords(res, num_landmarks):
    if res:
        return [val for lm in res.landmark for val in [lm.x, lm.y, lm.z]]
    else:
        return [0.0] * (num_landmarks * 3) 

def normalize_relative(pose,lh,rh):
    all_landmarks = np.concatenate([pose,lh,rh]).reshape(-1,3)
    origin = all_landmarks[0].copy()
    left_shoulder = all_landmarks[11]
    right_shoulder = all_landmarks[12]
    shoulder_width = np.linalg.norm(left_shoulder-right_shoulder)
    if shoulder_width<=0:
        shoulder_width=1
    all_landmarks = (all_landmarks - origin) / shoulder_width
    #TODO:IMPLEMENT  FOR FIX GHOST HANDS
    return all_landmarks.flatten()
if __name__ == "__main__":
    parse_and_extract()
