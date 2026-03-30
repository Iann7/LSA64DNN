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
    shifted_landmarks = video_landmarks.reshape(number_of_frames,-1,3)

    left_shoulder = shifted_landmarks[:,11,:]
    right_shoulder = shifted_landmarks[:,12,:]
    origin_coord = (left_shoulder + right_shoulder )/ 2 

    shoulder_width = np.linalg.norm(left_shoulder-right_shoulder,axis=1)
    shoulder_width[shoulder_width<=0] = 1.0

    shifted_landmarks = (shifted_landmarks - origin_coord[:, np.newaxis, :]) / shoulder_width[:, np.newaxis, np.newaxis]
    shifted_landmarks = shifted_landmarks.reshape(number_of_frames,-1)

    np.save(npy_path,shifted_landmarks)
    return 


if __name__ == "__main__":
    parse_and_extract()
