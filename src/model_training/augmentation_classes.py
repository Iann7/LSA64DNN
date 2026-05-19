import torch 
import numpy as np 
import  os
import constants as consts
from torchvision import transforms 

class AddNoise(object):
  def __init__(self,std=0.025):
     self.std = std 
  def __call__(self,tensor):
    noise = torch.randn_like(tensor) * self.std
    return tensor + noise

class TemporallyOccludeLandmarks(object):
   def __init__(self,drop_prob=0.05):
      self.drop_prob = drop_prob
   def __call__(self,tensor):
      frames = tensor.size(0)
      landmark_tensor = torch.reshape(tensor,(frames,consts.LANDMARK_SIZE,3))
      
      temporal_occlussion_mask = torch.rand(frames,consts.LANDMARK_SIZE)
      temporal_occlussion_mask = temporal_occlussion_mask > self.drop_prob
      temporal_occlussion_mask = temporal_occlussion_mask.float()
      # (pasamos de frames,consts.LANDMARK_SIZE) a (frames,consts.LANDMARK_SIZE,1)
      temporal_occlussion_mask = temporal_occlussion_mask.unsqueeze(-1)
      # (frames,consts.LANDMARK_SIZE,3) * (frames,consts.LANDMARK_SIZE,1)
      landmark_tensor = landmark_tensor * temporal_occlussion_mask
      # (frames,consts.LANDMARK_SIZE,3) -> (frames,225)
      tensor = torch.reshape(landmark_tensor,(frames,consts.INPUT_SIZE))
      return tensor

class CompletelyOccludeLandmarks(object):
   def __init__(self,drop_prob=0.25):
      self.drop_prob = drop_prob
   def __call__(self,tensor):
      frames = tensor.size(0)
      landmark_tensor = torch.reshape(tensor,(frames,consts.LANDMARK_SIZE,3))
      
      total_occlusion_mask = torch.rand(consts.LANDMARK_SIZE,1)
      total_occlusion_mask = total_occlusion_mask > self.drop_prob
      total_occlusion_mask = total_occlusion_mask.float()
      # (consts.LANDMARK_SIZE,1) -> (1,consts.LANDMARK_SIZE,1)
      total_occlusion_mask = total_occlusion_mask.unsqueeze(0)
      # (frames,consts.LANDMARK_SIZE,3) * (1,consts.LANDMARK_SIZE,1)
      landmark_tensor = landmark_tensor * total_occlusion_mask
      tensor = torch.reshape(landmark_tensor,(frames,consts.INPUT_SIZE))
      return tensor
class CompletelyOccludeBodyParts(object):

    def __init__(self, face_drop_prob=0.75, arm_drop_prob=0.5, leg_drop_prob=0.75):
        self.face_drop_prob = face_drop_prob
        self.arm_drop_prob = arm_drop_prob
        self.leg_drop_prob = leg_drop_prob
        self.face_indices = list(range(11))  
        self.left_arm_indices = [11, 13, 15, 17, 19, 21] 
        self.right_arm_indices = [12, 14, 16, 18, 20, 22]
        self.left_leg_indices = [23, 25, 27, 29, 31]
        self.right_leg_indices = [24, 26, 28, 30, 32]

    def __call__(self, tensor):
        frames = tensor.size(0)
        landmark_tensor = torch.reshape(tensor, (frames, consts.LANDMARK_SIZE, 3))
        mask = torch.ones(consts.LANDMARK_SIZE, 1)
        
        if torch.rand(1) < self.face_drop_prob:
            mask[self.face_indices] = 0
        
        if torch.rand(1) < self.arm_drop_prob:
            mask[self.left_arm_indices] = 0
        
        if torch.rand(1) < self.arm_drop_prob:
            mask[self.right_arm_indices] = 0
        
        if torch.rand(1) < self.leg_drop_prob:
            mask[self.left_leg_indices] = 0
        
        if torch.rand(1) < self.leg_drop_prob:
            mask[self.right_leg_indices] = 0
        
        mask_expanded = mask.unsqueeze(0)
        landmark_tensor = landmark_tensor * mask_expanded
        
        return torch.reshape(landmark_tensor, (frames, consts.INPUT_SIZE))

class TimeWarping(object):
    def __init__(self, sigma=0.15):
        self.sigma = sigma
    
    def __call__(self, tensor):
        frames = tensor.size(0)
        if frames < 10:
            return tensor
            
        t = np.linspace(0, 1, frames)
        warping = np.cumsum(np.random.normal(0, self.sigma, frames))
        warping = (warping - warping.min()) / (warping.max() - warping.min() + 1e-8)
        new_t = t + (warping - t) * self.sigma
        indices = np.interp(t, new_t, np.arange(frames))
        indices = np.clip(indices, 0, frames-1).astype(int)
        return tensor[indices]

class ScaleLandmarks(object):
    def __init__(self, min_scale=0.9, max_scale=1.1):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, tensor):
        # tensor shape: [frames, 225]
        frames = tensor.size(0)
        # 1. Reshape a [frames, 75, 3]
        landmark_tensor = torch.reshape(tensor,(frames,consts.LANDMARK_SIZE,3))
        
        scale_factor = torch.empty(1).uniform_(self.min_scale, self.max_scale).item()
        
        # 3. Aplicar el escalado SOLO a las coordenadas X e Y (y dejar Z tal vez)
        # Asumiendo que los 3 canales son [x, y, z] o [x, y, visibilidad]
        landmark_tensor[..., :2] = landmark_tensor[..., :2] * scale_factor
        
        # 4. Reshape de vuelta a [frames, 225]
        return torch.reshape(landmark_tensor,(frames,consts.INPUT_SIZE))
def init_transforms():
    train_transform = transforms.Compose([AddNoise(std=0.01)
                                          ,TemporallyOccludeLandmarks(drop_prob=0.08)
                                          #,CompletelyOccludeLandmarks(drop_prob=0.01)
                                          ,TimeWarping(sigma=0.05)
                                          ,ScaleLandmarks(min_scale=0.9, max_scale=1.1)
                                          #,CompletelyOccludeBodyParts(face_drop_prob=0.consts.LANDMARK_SIZE, arm_drop_prob=0.15, leg_drop_prob=0.consts.LANDMARK_SIZE)
                                          ])
    val_transform = None 
    test_transform = None
    return train_transform,val_transform,test_transform