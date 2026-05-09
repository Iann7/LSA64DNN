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
    num_workers = 7
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
        pose_model,hands_model = init_mediapipe()
        
        stem = video_path.stem  
        parts = stem.split('_')  
        sign_id = int(parts[0])
        
        video_landmarks = parse_video(video_path, pose_model,hands_model)
        pose_model.close()
        hands_model.close()

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
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,      
        model_complexity=1,           
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,      
        max_num_hands=2,
        model_complexity=1,           
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return pose, hands

def parse_video(video_path, pose,hands):
    cap = cv2.VideoCapture(str(video_path))
    landmarks = [] 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 3. Pass holistic through here
        landmark = extract_landmark_from_frame(frame,pose,hands)
        landmarks.append(landmark)
    cap.release()
    return landmarks

def extract_landmark_from_frame(frame,pose,hands):

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = get_pose_coords(pose.process(rgb_frame).pose_landmarks,33)
    hand_result = extract_hand_coords(hands.process(rgb_frame))
    return np.concatenate([pose_result,hand_result])# ✓ FIXED: iterate over the list

def get_pose_coords(res, num_landmarks):
    if res:
        return [val for lm in res.landmark for val in [lm.x, lm.y, lm.z]]
    else:
        return [0.0] * (num_landmarks * 3) 
    
def extract_hand_coords(hand_result):
    output_coords = [0.0] * (42 * 3) 
    
    if not hand_result.multi_hand_landmarks:
        return output_coords
        
    for idx, hand_handedness in enumerate(hand_result.multi_handedness):
        # MediaPipe a veces invierte las manos en el video, 
        # pero el label 'Left'/'Right' suele ser consistente con la anatomía.
        label = hand_handedness.classification[0].label # "Left" o "Right"
        landmarks = hand_result.multi_hand_landmarks[idx]
        
        temp_coords = []
        for lm in landmarks.landmark:
            temp_coords.extend([lm.x, lm.y, lm.z])
            
        if label == "Right":
            output_coords[0:63] = temp_coords  # Los primeros 21 puntos
        else:
            output_coords[63:126] = temp_coords # Los segundos 21 puntos
            
    return output_coords

if __name__ == "__main__":
    parse_and_extract()