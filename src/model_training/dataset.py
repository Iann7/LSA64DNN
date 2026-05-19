from augmentation_classes import AddNoise,TemporallyOccludeLandmarks,CompletelyOccludeLandmarks,TimeWarping
import torch 
import numpy as np 
import os
import pandas as pd 
from torch.utils.data import Dataset,DataLoader,Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms 
from torch.utils.data import Dataset,DataLoader,random_split
from augmentation_classes import init_transforms
import torch.nn.functional as F
import constants as const

# Paths
DATA_DIR = "data/raw"
POSE_DIR = "data/poses/shifted_and_blurred"
METADATA_DIR = "data/metadata/real_labels.csv"
class LSA64Dataset(Dataset):
  def __init__(self,data,data_dir,max_frames=150,transform=None,indices=None):
    self.data = data
    self.data_dir = data_dir
    self.max_frames = max_frames
    self.transform = transform 
    self.indices = indices if indices is not None else range(len(data))
    
  def __len__(self):    
    return len(self.indices)

  def __getitem__(self,idx):
    real_idx = self.indices[idx]
    sign_pose = np.load(os.path.join(self.data_dir,self.data.loc[real_idx,'filename']))
    sign_pose = torch.FloatTensor(sign_pose)
    sign_pose = self.augment_landmarks(sign_pose)
    label = self.data.loc[real_idx,'sign_id'] -1
    return (sign_pose,torch.tensor(label,dtype=torch.long)) 

  def augment_landmarks(self, sign_pose):
      if self.transform:
          sign_pose = self.transform(sign_pose)
      return sign_pose

  def pad_frames(self, sign_pose):
      current_frames = sign_pose.shape[0]

      if current_frames < self.max_frames:
          frames_to_pad = self.max_frames - current_frames
          padding_config = (0, 0, 0, frames_to_pad)
          sign_pose = F.pad(sign_pose, padding_config, mode='constant', value=0)

      elif current_frames > self.max_frames:
          sign_pose = sign_pose[:self.max_frames, :]

      return sign_pose

def collate_fn_with_lengths(batch):
    signs, labels = zip(*batch)
    
    lengths = torch.tensor([sign.size(0) for sign in signs])
    
    max_len = lengths.max().item()
    padded_signs = torch.zeros(len(signs), max_len, signs[0].size(1))
    
    for i, sign in enumerate(signs):
        padded_signs[i, :sign.size(0), :] = sign
    
    return padded_signs, torch.tensor(labels), lengths

def split_dataset():
    train_transform, val_transform, test_transform = init_transforms()

    full_dataset = LSA64Dataset(pd.read_csv(METADATA_DIR),DATA_DIR,max_frames=120,transform=None)
    
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(2026) #IMPORTANTE,mantenemos la misma seed siempre
    train_subset, val_subset, test_subset = random_split(full_dataset, [train_size, val_size, test_size],generator=generator)

    train_subset.transform = train_transform
    val_subset.transform = val_transform
    test_subset.transform = test_transform

    train_loader, val_loader, test_loader = initialize_loaders(train_subset, val_subset, test_subset)
    return train_loader,val_loader,test_loader

def initialize_loaders(train_subset, val_subset, test_subset):
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn_with_lengths)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True,collate_fn=collate_fn_with_lengths)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True,collate_fn=collate_fn_with_lengths)
    return train_loader,val_loader,test_loader

def split_dataset_by_LOSO(signer_id):
    train_transform, val_transform, test_transform = init_transforms()

    dataframe = pd.read_csv(METADATA_DIR)
    signer_ids = dataframe['subject'].values
    
    train_indices = [i for i, sid in enumerate(signer_ids) if sid != signer_id]
    test_indices = [i for i, sid in enumerate(signer_ids) if sid == signer_id]
    
    
    train_idx,val_idx = train_test_split(train_indices, test_size=0.1, random_state=2026)
    
    train_dataset = LSA64Dataset(dataframe, POSE_DIR, max_frames=120, transform=train_transform, indices=train_idx)
    val_dataset = LSA64Dataset(dataframe, POSE_DIR, max_frames=120, transform=val_transform, indices=val_idx)
    test_dataset = LSA64Dataset(dataframe, POSE_DIR, max_frames=120, transform=test_transform, indices=test_indices)

    train_loader,val_loader,test_loader =initialize_loaders(train_dataset, val_dataset, test_dataset)
    
    return train_loader, val_loader, test_loader
   