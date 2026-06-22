#!/usr/bin/env python
"""
Spatial Placement Training Script
---------------------------------
This script trains a neural network (CNN-LSTM, LSTM, or Transformer)
to predict the spatial positions (x, y) of hit notes using polar delta
coordinates (relative angle and distance changes).
"""

import os
import sys
import argparse
import time
import traceback
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Add project directories to sys.path to support imports under the new layout
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
rhythm_dir = os.path.join(project_root, 'rhythm')
if rhythm_dir not in sys.path:
    sys.path.append(rhythm_dir)
spatial_dir = os.path.join(project_root, 'spatial')
if spatial_dir not in sys.path:
    sys.path.append(spatial_dir)

# Our imports
from spatial_model import CNNLSTMSpatialModel, LSTMSpatialModel, TransformerSpatialModel
from train_rhythm import OsuBeatmapParser

class OsuBeatmapSpatialParser:
    """Helper class to parse Osu! beatmap files for spatial coordinates."""
    @staticmethod
    def parse(filepath):
        hit_objects = []
        if not os.path.exists(filepath):
            return hit_objects
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        sections = {}
        current_section = None
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)
                
        # Parse Hit Objects
        if 'HitObjects' in sections:
            for line in sections['HitObjects']:
                parts = line.split(',')
                if len(parts) < 5:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    time_ms = float(parts[2])
                    obj_type = int(parts[3])
                    
                    is_circle = bool(obj_type & 1)
                    is_slider = bool(obj_type & 2)
                    is_spinner = bool(obj_type & 8)
                    
                    if is_circle:
                        t_class = 0 # Circle
                    elif is_slider:
                        t_class = 1 # Slider
                    elif is_spinner:
                        t_class = 2 # Spinner
                    else:
                        t_class = 3 # Other
                        
                    hit_objects.append({
                        'x': x,
                        'y': y,
                        'time': time_ms,
                        'class': t_class
                    })
                except (ValueError, IndexError):
                    continue
                    
        # Sort hit objects by time
        hit_objects.sort(key=lambda x: x['time'])
        return hit_objects

class OsuSpatialDataset(Dataset):
    """Custom Dataset for sequence prediction of polar delta coordinates."""
    def __init__(self, maps_dir, spec_dir, sr=11025, hop_length=512, chunk_size=64, is_train=True, mapset_ids=None):
        self.maps_dir = maps_dir
        self.spec_dir = spec_dir
        self.sr = sr
        self.hop_length = hop_length
        self.chunk_size = chunk_size
        self.is_train = is_train
        self.mapset_ids = mapset_ids
        
        self.valid_pairs = []
        self._find_valid_pairs()
        
    def _find_valid_pairs(self):
        if not os.path.exists(self.maps_dir):
            print(f"Maps directory '{self.maps_dir}' does not exist.")
            return
            
        map_files = [f for f in os.listdir(self.maps_dir) if f.endswith('.osu')]
        print(f"Scanning {len(map_files)} maps for spatial data...")
        
        for map_file in map_files:
            mapset_id = map_file.split('_')[0]
            spec_file = f"{mapset_id}.npy"
            spec_path = os.path.join(self.spec_dir, spec_file)
            
            if os.path.exists(spec_path):
                if self.mapset_ids is not None and mapset_id not in self.mapset_ids:
                    continue
                map_path = os.path.join(self.maps_dir, map_file)
                # Parse hit objects immediately (lightweight)
                hit_objs = OsuBeatmapSpatialParser.parse(map_path)
                if len(hit_objs) >= 5: # Needs at least a few objects
                    self.valid_pairs.append({
                        'map_path': map_path,
                        'spec_path': spec_path,
                        'hit_objects': hit_objs,
                        'filename': map_file,
                        'mapset_id': mapset_id
                    })
                    
        print(f"Found {len(self.valid_pairs)} valid spatial map-spectrogram pairs.")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        hit_objs = pair['hit_objects']
        spec_path = pair['spec_path']
        
        # Load spectrogram
        S = np.load(spec_path)
        n_bins, n_frames = S.shape
        
        # Parse difficulty metadata and calculate density
        meta = OsuBeatmapParser.parse_metadata(pair['map_path'])
        song_duration_sec = (n_frames * self.hop_length / self.sr)
        density = len(hit_objs) / song_duration_sec if song_duration_sec > 0.0 else 0.0
        
        meta_vec = np.array([
            meta['hp'],
            meta['cs'],
            meta['od'],
            meta['ar'],
            meta['sm'],
            meta['str'],
            density
        ], dtype=np.float32)
        
        # Select sequence start index
        N = len(hit_objs)
        if N > self.chunk_size:
            if self.is_train:
                start_idx = np.random.randint(0, N - self.chunk_size)
            else:
                start_idx = 0
            selected_objs = hit_objs[start_idx : start_idx + self.chunk_size]
            
            # Get the object just before the chunk to calculate previous angle/movement vector
            if start_idx > 0:
                prev_obj = hit_objs[start_idx - 1]
                # Also need the one before prev_obj to estimate previous angle
                if start_idx > 1:
                    prev_prev = hit_objs[start_idx - 2]
                    prev_angle = math.atan2(prev_obj['y'] - prev_prev['y'], prev_obj['x'] - prev_prev['x'])
                else:
                    prev_angle = 0.0
            else:
                prev_obj = None
                prev_angle = 0.0
        else:
            selected_objs = hit_objs
            prev_obj = None
            prev_angle = 0.0
            
        seq_len = len(selected_objs)
        
        # Build features and targets
        features = []
        targets = []
        
        # Track coordinates and angles
        curr_prev_obj = prev_obj
        curr_prev_angle = prev_angle
        
        for i in range(seq_len):
            obj = selected_objs[i]
            
            # 1. Timing delta: log(1 + delta_t)
            if curr_prev_obj is not None:
                delta_t = obj['time'] - curr_prev_obj['time']
                prev_x, prev_y = curr_prev_obj['x'], curr_prev_obj['y']
            else:
                delta_t = 500.0  # Default initial delta
                prev_x, prev_y = 256.0, 192.0  # Default screen center
                
            delta_t_feat = math.log(1.0 + max(0.0, delta_t))
            
            # 2. Note type one-hot
            type_feat = [0.0] * 4
            type_feat[obj['class']] = 1.0
            
            # 3. Audio context: 5 frames centered at note time
            f_center = int(round(obj['time'] * self.sr / (self.hop_length * 1000.0)))
            
            audio_frames = []
            for offset in [-2, -1, 0, 1, 2]:
                f_idx = f_center + offset
                if 0 <= f_idx < n_frames:
                    audio_frames.append(S[:, f_idx])
                else:
                    audio_frames.append(np.zeros(n_bins))
            audio_feat = np.concatenate(audio_frames) # 5 * 84 = 420 dims
            
            # Stack input feature: [delta_t_feat (1), type_feat (4), audio_feat (420), meta_vec (7)] -> 432 dims
            feat = np.concatenate([[delta_t_feat], type_feat, audio_feat, meta_vec])
            features.append(feat)
            
            # 4. Target polar coordinates: sin(delta_theta), cos(delta_theta), distance
            dx = obj['x'] - prev_x
            dy = obj['y'] - prev_y
            dist = math.sqrt(dx*dx + dy*dy)
            
            theta = math.atan2(dy, dx)
            delta_theta = theta - curr_prev_angle
            # Normalize to [-pi, pi]
            delta_theta = (delta_theta + math.pi) % (2 * math.pi) - math.pi
            
            target = [math.sin(delta_theta), math.cos(delta_theta), dist]
            targets.append(target)
            
            # Update history
            curr_prev_angle = theta
            curr_prev_obj = obj
            
        features = np.array(features, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        # Padding if sequence is too short
        actual_len = features.shape[0]
        if actual_len < self.chunk_size:
            pad_len = self.chunk_size - actual_len
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
            # For targets, pad with cos=1, sin=0, dist=0
            targets_pad = np.zeros((pad_len, 3), dtype=np.float32)
            targets_pad[:, 1] = 1.0 # cos(0) = 1.0
            targets = np.vstack([targets, targets_pad])
            
        return torch.tensor(features), torch.tensor(targets), actual_len

def loss_fn(pred, target, actual_lens):
    """Custom loss function masking out padded values."""
    # pred, target shape: (batch_size, chunk_size, 3)
    # Target structure: [sin, cos, distance]
    batch_size, chunk_size, _ = pred.shape
    
    # Create mask for sequence lengths
    mask = torch.zeros(batch_size, chunk_size, device=pred.device)
    for b in range(batch_size):
        mask[b, :actual_lens[b]] = 1.0
        
    mask = mask.unsqueeze(-1) # shape: (batch_size, chunk_size, 1)
    
    # 1. Direction loss (Mean Squared Error on sin/cos predictions)
    # Also enforcing unit length constraint: sin^2 + cos^2 = 1 (helps stabilize predictions)
    sin_pred, cos_pred = pred[:, :, 0:1], pred[:, :, 1:2]
    sin_target, cos_target = target[:, :, 0:1], target[:, :, 1:2]
    
    dir_loss = torch.sum(((sin_pred - sin_target)**2 + (cos_pred - cos_target)**2) * mask) / (torch.sum(mask) + 1e-8)
    
    # Unit vector regularization
    norm_reg = torch.sum(((sin_pred**2 + cos_pred**2 - 1.0)**2) * mask) / (torch.sum(mask) + 1e-8)
    
    # 2. Distance loss (MSE on distance)
    # Using Softplus/ReLU inside evaluation, raw predicting here. Target distance is raw.
    dist_pred = pred[:, :, 2:3]
    dist_target = target[:, :, 2:3]
    
    # Softplus activation on predicted distance to guarantee positive outputs
    dist_pred_soft = torch.nn.functional.softplus(dist_pred)
    dist_loss = torch.sum(((dist_pred_soft - dist_target)**2) * mask) / (torch.sum(mask) + 1e-8)
    
    # Combine losses (balance coefficient 0.001 because distance is in pixels [0, 500], while sin/cos are in [-1, 1])
    total_loss = dir_loss + 0.1 * norm_reg + 0.0001 * dist_loss
    
    return total_loss, dir_loss.item(), dist_loss.item()

def train_epoch(model, dataloader, optimizer, device, teacher_forcing_ratio=1.0):
    model.train()
    total_loss = 0
    total_dir_loss = 0
    total_dist_loss = 0
    
    for features, targets, lengths in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        if isinstance(model, TransformerSpatialModel):
            preds = model(features)
        else:
            preds = model(features, targets=targets, teacher_forcing_ratio=teacher_forcing_ratio)
        
        loss, dir_l, dist_l = loss_fn(preds, targets, lengths)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_dir_loss += dir_l
        total_dist_loss += dist_l
        
    n = len(dataloader)
    return total_loss / n, total_dir_loss / n, total_dist_loss / n

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_dir_loss = 0
    total_dist_loss = 0
    
    all_dist_errors = []
    all_angle_errors = []
    
    with torch.no_grad():
        for features, targets, lengths in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            
            preds = model(features)
            loss, dir_l, dist_l = loss_fn(preds, targets, lengths)
            
            total_loss += loss.item()
            total_dir_loss += dir_l
            total_dist_loss += dist_l
            
            # Calculate readable errors (pixels and degrees)
            for b in range(features.shape[0]):
                act_len = lengths[b].item()
                p_dist = torch.nn.functional.softplus(preds[b, :act_len, 2]).cpu().numpy()
                t_dist = targets[b, :act_len, 2].cpu().numpy()
                
                # Distance error in pixels
                all_dist_errors.extend(np.abs(p_dist - t_dist))
                
                # Angle error in degrees
                p_sin = preds[b, :act_len, 0].cpu().numpy()
                p_cos = preds[b, :act_len, 1].cpu().numpy()
                t_sin = targets[b, :act_len, 0].cpu().numpy()
                t_cos = targets[b, :act_len, 1].cpu().numpy()
                
                p_angle = np.arctan2(p_sin, p_cos)
                t_angle = np.arctan2(t_sin, t_cos)
                
                angle_diff = np.abs(p_angle - t_angle)
                angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
                all_angle_errors.extend(np.degrees(angle_diff))
                
    n = len(dataloader)
    mean_dist_err = np.mean(all_dist_errors) if all_dist_errors else 0.0
    mean_ang_err = np.mean(all_angle_errors) if all_angle_errors else 0.0
    
    return total_loss / n, total_dir_loss / n, total_dist_loss / n, mean_dist_err, mean_ang_err

def main():
    parser = argparse.ArgumentParser(description="Train spatial note placement model using polar delta coordinates.")
    parser.add_argument("--maps_dir", type=str, default="maps", help="Maps directory path")
    parser.add_argument("--spec_dir", type=str, default="spectrograms", help="Spectrograms directory path")
    parser.add_argument("--model_type", type=str, choices=["cnn-lstm", "lstm", "transformer"], default="cnn-lstm", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation dataset split ratio")
    parser.add_argument("--save_path", type=str, default="spatial_model.pth", help="Model checkpoint save path")
    parser.add_argument("--chunk_size", type=int, default=64, help="Sequence length of notes for training")
    parser.add_argument("--hop_length", type=int, default=512, help="Spectrogram hop length (default: 512)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Split train/val by unique mapset IDs to prevent leakage of different difficulties of the same song
    map_files = [f for f in os.listdir(args.maps_dir) if f.endswith('.osu')]
    all_mapset_ids = set()
    for map_file in map_files:
        mapset_id = map_file.split('_')[0]
        spec_file = f"{mapset_id}.npy"
        if os.path.exists(os.path.join(args.spec_dir, spec_file)):
            all_mapset_ids.add(mapset_id)
            
    all_mapset_ids = sorted(list(all_mapset_ids))
    if len(all_mapset_ids) == 0:
        print("No training data found. Make sure maps (.osu) and spectrograms (.npy) exist.")
        sys.exit(1)
        
    np.random.seed(42) # set seed for reproducibility
    np.random.shuffle(all_mapset_ids)
    val_count = int(len(all_mapset_ids) * args.val_split)
    val_mapset_ids = set(all_mapset_ids[:val_count])
    train_mapset_ids = set(all_mapset_ids[val_count:])
    
    print(f"Split {len(all_mapset_ids)} unique songs into {len(train_mapset_ids)} train and {len(val_mapset_ids)} validation songs.")
    
    train_dataset = OsuSpatialDataset(
        maps_dir=args.maps_dir,
        spec_dir=args.spec_dir,
        chunk_size=args.chunk_size,
        is_train=True,
        mapset_ids=train_mapset_ids,
        hop_length=args.hop_length
    )
    val_dataset = OsuSpatialDataset(
        maps_dir=args.maps_dir,
        spec_dir=args.spec_dir,
        chunk_size=args.chunk_size,
        is_train=False,
        mapset_ids=val_mapset_ids,
        hop_length=args.hop_length
    )
    
    if len(train_dataset) == 0:
        print("Train dataset is empty!")
        sys.exit(1)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Set up model and determine input dimension dynamically
    input_dim = train_dataset[0][0].shape[1]
    
    if args.model_type == "cnn-lstm":
        model = CNNLSTMSpatialModel(input_dim=input_dim).to(device)
    elif args.model_type == "lstm":
        model = LSTMSpatialModel(input_dim=input_dim).to(device)
    else:
        model = TransformerSpatialModel(input_dim=input_dim).to(device)
        
    print(f"Initialized model ({args.model_type}) with input dimension {input_dim}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine Annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print("-" * 80)
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Linearly decay teacher forcing ratio from 1.0 to 0.2
        tf_ratio = max(0.2, 1.0 - (epoch - 1) / args.epochs)
        
        train_loss, train_dir, train_dist = train_epoch(model, train_loader, optimizer, device, teacher_forcing_ratio=tf_ratio)
        val_loss, val_dir, val_dist, mean_dist_err, mean_ang_err = evaluate(model, val_loader, device)
        
        scheduler.step()
        elapsed = time.time() - start_time
        
        # Print epoch summary
        print(f"Epoch {epoch:2d}/{args.epochs:2d} | "
              f"Train Loss: {train_loss:.4f} (Dir: {train_dir:.4f}, Dist: {train_dist:.1f}) | "
              f"Val Loss: {val_loss:.4f} (Dir: {val_dir:.4f}, Dist: {val_dist:.1f}) | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {elapsed:.1f}s")
        print(f"   -> Val Errors: Avg Dist Error = {mean_dist_err:.2f} pixels, Avg Angle Error = {mean_ang_err:.2f} degrees")
        
        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"   [*] Saved new best model checkpoint to '{args.save_path}'")
            
    print("-" * 80)
    print("Training complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
