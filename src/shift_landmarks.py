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
OUTPUT_DIR = Path("data/poses/shifted")
METADATA_DIR = Path("data/metadata")
POSE_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# NPY Files
npy_files = list(POSE_DIR.glob("*.npy")) 



def parse_and_extract():
    num_workers = 2
    print(f"Starting parallel processing with {num_workers} cores!")
    print(len(npy_files))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_video = {executor.submit(process_single_npy, npy_path): npy_path for npy_path in npy_files}
        for future in tqdm(as_completed(future_to_video),total=len(npy_files),desc="Processing Videos"):
            result = future.result()

def process_single_npy(npy_path):
    video_landmarks = np.load(npy_path)
    number_of_frames = video_landmarks.shape[0]
    print( f"min {video_landmarks.min()} and max {video_landmarks.max()}")
    shifted_landmarks = video_landmarks.reshape(number_of_frames,-1,3)
    left_shoulder = shifted_landmarks[:,11,:]
    right_shoulder = shifted_landmarks[:,12,:]

    left_hip = shifted_landmarks[0,23,:]
    right_hip = shifted_landmarks[0,24,:]

    center_shoulder = (left_shoulder + right_shoulder )/ 2
    center_hip = (left_hip + right_hip )/ 2

    origin_coord = center_shoulder

    torso_width = np.linalg.norm(center_shoulder[0]-center_hip)
    #torso_width[torso_width<=0] = 1.0

    shifted_landmarks = (shifted_landmarks - origin_coord[:, np.newaxis, :]) / torso_width
    shifted_landmarks = shifted_landmarks.reshape(number_of_frames,-1)
    output_path = OUTPUT_DIR / npy_path.name
    np.save(output_path,shifted_landmarks)
    return 


if __name__ == "__main__":
    parse_and_extract()
