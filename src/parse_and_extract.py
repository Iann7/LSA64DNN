from pathlib import Path
import mediapipe as mp
import cv2  
import numpy as np 
import re 
import pandas as pd 
from tqdm import tqdm 

# Paths
DATA_DIR = Path("data/raw")
POSE_DIR = Path("data/poses/real")
METADATA_DIR = Path("data/metadata")
POSE_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Video Files
video_files = list(DATA_DIR.glob("*.mp4")) 

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)

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
    all_labels = []
    for video_path in tqdm(video_files,desc="Processing Videos"):
        stem = video_path.stem  
        parts = stem.split('_')  
        sign_id = int(parts[0])
        subject = int(parts[1])
        repetition = int(parts[2])
        sign_name = SIGN_NAMES[sign_id]  
        video_landmarks = parse_video(video_path)
        if len(video_landmarks) > 0:
            poses_array = np.array(video_landmarks)
            output_path = POSE_DIR / f"{stem}.npy"
            np.save(output_path,poses_array)
            all_labels.append({
                'filename': f"{stem}.npy",
                'sign_id': sign_id,
                'subject': subject,
                'repetition': repetition,
                'sign_name': sign_name,
                'num_frames': len(video_landmarks)
            })
            print(f"Saved {stem}:{poses_array.shape}")
    if all_labels:
        data_frame = pd.DataFrame(all_labels)
        data_frame.to_csv(METADATA_DIR / 'real_labels.csv',index=False)
        print(f"\n Saved labels to {METADATA_DIR / 'real_labels.csv'}")  
        print(f"   Total: {len(data_frame)} videos processed")                     

def parse_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    landmarks = [] 
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        landmark = extract_landmark_from_frame(frame)
        if landmark:
            landmarks.append(landmark)
    cap.release()
    return landmarks

def extract_landmark_from_frame(frame):
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = mp_pose.process(rgb_frame)
    if result.pose_landmarks:
        landmarks = []
        for lm in result.pose_landmarks.landmark:
            landmarks.extend([lm.x,lm.y,lm.z]) 
        return landmarks
    else:
        return None 

if __name__ == "__main__":
    parse_and_extract()