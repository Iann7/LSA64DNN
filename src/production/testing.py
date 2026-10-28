import sys
from pathlib import Path
import time
from collections import deque
import numpy as np 
import cv2
import torch
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model_training.model import LSA64Classifier, load_model_with_best_weights
from src.pose_extraction.parse_and_extract import init_mediapipe, extract_landmark_from_frame
from src.pose_extraction.shift_landmarks import process_sequence_pose

USE_GPU            = True
CAM_INDEX          = 0
DRAW_POSE          = True
DRAW_HANDS         = True
WINDOW             = 30
INFERENCE_INTERVAL = 30

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

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15),
    (12, 14), (14, 16), (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

class WebcamStream:
    def __init__(self, src=0, width=480, height=360):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                break
            with self.lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else (False, None)

    def stop(self):
        self.stopped = True
        self.stream.release()

def draw_pose_flat(frame, pose_coords):
    if not DRAW_POSE or len(pose_coords) < 99:
        return
    h, w = frame.shape[:2]
    pts = pose_coords.reshape(-1, 3)
    pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in pts]
    for a, b in POSE_CONNECTIONS:
        if a < len(pixel_pts) and b < len(pixel_pts):
            cv2.line(frame, pixel_pts[a], pixel_pts[b], (255, 200, 0), 2)
    for p in pixel_pts:
        cv2.circle(frame, p, 3, (0, 128, 255), -1)

def draw_hand_flat(frame, hand_coords, color=(0, 255, 0)):
    if not DRAW_HANDS or len(hand_coords) < 63:
        return
    h, w = frame.shape[:2]
    pts = hand_coords.reshape(-1, 3)
    pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in pts]
    if np.all(pts == 0):
        return
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pixel_pts[a], pixel_pts[b], color, 2)
    for p in pixel_pts:
        cv2.circle(frame, p, 3, color, -1)

def feed_into_model(model, extracted_poses, device="cpu"):
    if len(extracted_poses) < WINDOW:
        return None
    sequence_array = np.array(list(extracted_poses))
    sequence_array = process_sequence_pose(sequence_array)
    x = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0).to(device)
    lengths = torch.tensor([WINDOW], dtype=torch.long, device=device)
    with torch.no_grad():
        return model(x, lengths)

def main():
    device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    LSA_model = load_model_with_best_weights()
    LSA_model.to(device)
    LSA_model.eval()

    pose, hands = init_mediapipe() 

    # Start multi-threaded capture stream
    vs = WebcamStream(src=CAM_INDEX, width=480, height=360).start()
    time.sleep(1.0) # Warm up camera thread

    print("[info] Press 'q' to quit.")

    prev_time = time.time()
    previous_poses = deque(maxlen=WINDOW)
    current_prediction_text = "Waiting for frames..."
    frame_count = 0

    try:
        while True:
            ok, frame = vs.read()
            if not ok or frame is None:
                print("[warn] Frame capture failed.")
                break

            # 1. Extract raw landmarks
            landmarks_vector = extract_landmark_from_frame(frame, pose, hands)
            previous_poses.append(landmarks_vector)

            # 2. Conditional Inference
            frame_count += 1
            if len(previous_poses) == WINDOW and (frame_count % INFERENCE_INTERVAL == 0):
                sequence_array = np.array(list(previous_poses))
                sequence_array = process_sequence_pose(sequence_array)
                
                x = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0).to(device)
                lengths = torch.tensor([WINDOW], dtype=torch.long, device=device)
                with torch.no_grad():
                    pred_logits = LSA_model(x, lengths)
                
                if pred_logits is not None:
                    predicted_class = torch.argmax(pred_logits, dim=1).item()
                    current_prediction_text = SIGN_NAMES.get(predicted_class + 1, f"Sign ID {predicted_class}")

            # 3. Drawing using the raw/un-shifted landmarks so they align with the camera frame
            draw_pose_flat(frame, landmarks_vector[:99])
            draw_hand_flat(frame, landmarks_vector[99:162], color=(0, 255, 0))
            draw_hand_flat(frame, landmarks_vector[162:225], color=(255, 0, 255))

            # HUD
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Prediction: {current_prediction_text}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Sign Language RT Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        vs.stop()
        cv2.destroyAllWindows()
        pose.close()
        hands.close()

if __name__ == "__main__":
    main()