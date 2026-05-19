from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.ndimage import gaussian_filter1d
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
OUTPUT_DIR = Path("data/poses/shifted_and_blurred")
METADATA_DIR = Path("data/metadata")
POSE_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# NPY Files
npy_files = list(POSE_DIR.glob("*.npy")) 



def parse_and_shift():
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

    fill_in_zeroes(shifted_landmarks)
    shifted_landmarks = shift_landmarks(shifted_landmarks)
    apply_gaussian_filter(shifted_landmarks)

    shifted_landmarks = shifted_landmarks.reshape(number_of_frames,-1)
    output_path = OUTPUT_DIR / npy_path.name
    np.save(output_path,shifted_landmarks)
    return 

def fill_in_zeroes(shifted_landmarks):
    num_landmarks = shifted_landmarks.shape[1]
    for dim in range(3):
        for landmark in range(num_landmarks):
            shifted_landmarks[:,landmark,dim] =  interpolate1D(shifted_landmarks[:,landmark,dim])
def interpolate1D(array):
    valid_idx = np.where(array!=0)[0]
    if len(valid_idx) == 0:
        return array
    all_idx = np.arange(len(array))
    return np.interp(all_idx,valid_idx,array[valid_idx])
def apply_gaussian_filter(shifted_landmarks):
    #Apply the gaussian filter temporally for each landmark separately for each dimension of that landmark (XYZ) 
    for dim in range(3):
        shifted_landmarks[:,:,dim] = gaussian_filter1d(shifted_landmarks[:,:,dim],sigma=5.0,axis=0)

def shift_landmarks(landmarks):
    left_shoulder = landmarks[:,11,:]
    right_shoulder = landmarks[:,12,:]

    left_hip = landmarks[0,23,:]
    right_hip = landmarks[0,24,:]

    center_shoulder = (left_shoulder + right_shoulder )/ 2
    center_hip = (left_hip + right_hip )/ 2

    origin_coord = center_shoulder

    torso_width = np.linalg.norm(center_shoulder[0]-center_hip)
    # origin_coord debe tener shape (frames, 1, 3) para que numpy haga el broadcast bien
    shifted_landmarks =  (landmarks - origin_coord[:, np.newaxis, :]) / (torso_width + 1e-6)
    #left_hand  =     shift_and_normalize_hand(landmarks[:,33:55,:])
    #right_hand =    shift_and_normalize_hand(landmarks[:,55:76,:])
    return shifted_landmarks



def shift_and_normalize_hand(hand):
    wrist = hand[:, 0:1, :]  
    palm_width = np.linalg.norm(hand[:, 9, :] - hand[:, 0, :], axis=1, keepdims=True)
    shifted_and_normalized_hand = (hand - wrist) / (palm_width[:, np.newaxis] + 1e-6)
    return shifted_and_normalized_hand


def enforce_bone_lengths(original_bone_lengths,origin_bone_points,end_bone_points):
    current_directions = end_bone_points-origin_bone_points
    current_bone_lengths = np.linalg.norm(current_directions,axis=1,keepdims=True)
    mask = current_bone_lengths>1e-6
    scale = np.where(mask,original_bone_lengths/current_bone_lengths,1.0)
    return np.where(mask,origin_bone_points+(current_directions*scale),end_bone_points)


def get_stable_len(landmarks,p1, p2):
    lens = np.linalg.norm(landmarks[:, p1, :] - landmarks[:, p2, :], axis=1)  
    return np.percentile(lens, 90)
if __name__ == "__main__":
    parse_and_shift()
