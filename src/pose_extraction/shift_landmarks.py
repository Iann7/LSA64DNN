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
    shifted_landmarks =process_sequence_pose(video_landmarks)
    output_path = OUTPUT_DIR / npy_path.name
    np.save(output_path,shifted_landmarks)
    return 

import numpy as np

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        min_cutoff: Frecuencia de corte mínima (reduce ruido estático).
        beta: Parámetro de velocidad (reduce el retardo en movimiento).
        d_cutoff: Frecuencia de corte para la derivada.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_previous = x0
        self.dx_previous = np.zeros_like(x0)
        self.t_previous = t0

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, x):
        t_e = t - self.t_e_previous if hasattr(self, 't_e_previous') else 1.0 / 30.0 # Asumiendo 30fps por defecto si no hay delta
        # Simplificación para el primer frame:
        if self.t_previous is None or t == self.t_previous:
            dt = 1.0 / 30.0
        else:
            dt = t - self.t_previous

        self.t_previous = t

        # Derivada estimada
        dx = (x - self.x_previous) / dt
        edx = self._alpha(self.d_cutoff, dt) * dx + (1.0 - self._alpha(self.d_cutoff, dt)) * self.dx_previous
        self.dx_previous = edx

        # Frecuencia de corte adaptativa basada en la velocidad
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        alpha = self._alpha(cutoff, dt)
        
        # Filtrado exponencial
        x_filtered = alpha * x + (1.0 - alpha) * self.x_previous
        self.x_previous = x_filtered
        return x_filtered




def apply_one_euro_filter(shifted_landmarks, min_cutoff=1.7, beta=0.03):
    n_frames, n_landmarks, n_dims = shifted_landmarks.shape
    filtered_landmarks = np.copy(shifted_landmarks)
    
    # Inicializar el filtro con el primer frame
    # Usamos un tiempo simulado con dt = 1/30 (asumiendo 30 FPS)
    dt = 1.0 / 30.0
    x_prev = shifted_landmarks[0]
    dx_prev = np.zeros_like(x_prev)
    
    # Parámetros internos del filtro
    tau = 1.0 / (2.0 * np.pi * min_cutoff)
    
    for i in range(1, n_frames):
        x = shifted_landmarks[i]
        
        # Cálculo de la derivada (velocidad)
        dx = (x - x_prev) / dt
        # alpha para la derivada
        alpha_d = 1.0 / (1.0 / (2.0 * np.pi * 1.0 * dt) + 1.0) # d_cutoff por defecto en 1.0
        edx = alpha_d * dx + (1.0 - alpha_d) * dx_prev
        dx_prev = edx
        
        # Frecuencia de corte adaptativa por landmark y dimensión
        cutoff = min_cutoff + beta * np.abs(edx)
        tau_dynamic = 1.0 / (2.0 * np.pi * cutoff)
        alpha = 1.0 / (tau_dynamic / dt + 1.0)
        
        # Aplicar filtro
        x_filtered = alpha * x + (1.0 - alpha) * x_prev
        filtered_landmarks[i] = x_filtered
        x_prev = x_filtered
        
    return filtered_landmarks


def process_sequence_pose(poses):
    number_of_poses = poses.shape[0]
    shifted_landmarks = poses.reshape(number_of_poses, -1, 3)

    fill_in_zeroes(shifted_landmarks)
    shifted_landmarks = shift_landmarks(shifted_landmarks)
    
    shifted_landmarks = apply_one_euro_filter(shifted_landmarks)
    
    return shifted_landmarks.reshape(number_of_poses,-1)

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
