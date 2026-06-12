#!/usr/bin/env python
"""
Rhythm Generation (Temporal Transcription) Training Script
----------------------------------------------------------
This script parses Osu! beatmaps (.osu) and aligns them with Mel spectrograms (.npy)
to train a neural network (CNN-LSTM or Transformer) to transcribe rhythm.

Event Classes:
  0: No Event
  1: Circle Onset
  2: Slider Start
  3: Slider End
  4: Spinner Onset
"""

import os
import sys
import argparse
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

# Our imports
from rhythm_model import CNNLSTMRhythmModel, TransformerRhythmModel, CNNTransformerRhythmModel, load_model_helper

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_p = torch.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)
        
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1.0 - p_t) ** self.gamma
        
        if self.label_smoothing > 0.0:
            c = inputs.size(-1)
            smoothed_targets = torch.full_like(log_p, self.label_smoothing / (c - 1))
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            loss = -smoothed_targets * log_p
            loss = loss.sum(dim=-1) * focal_weight
        else:
            loss = -focal_weight * log_p_t
            
        if self.weight is not None:
            class_weights_gathered = self.weight.gather(0, targets)
            loss = loss * class_weights_gathered
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class OsuBeatmapParser:
    """Helper class to parse Osu! beatmap files."""
    @staticmethod
    def parse_metadata(filepath):
        # Default difficulty metadata
        meta = {
            'hp': 5.0,
            'cs': 4.0,
            'od': 5.0,
            'ar': 5.0,
            'sm': 1.4,
            'str': 1.0
        }
        if not os.path.exists(filepath):
            return meta
            
        try:
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
                    
            if 'Difficulty' in sections:
                for line in sections['Difficulty']:
                    if ':' in line:
                        key, val = line.split(':', 1)
                        k = key.strip()
                        try:
                            v = float(val.strip())
                            if k == 'HPDrainRate':
                                meta['hp'] = v
                            elif k == 'CircleSize':
                                meta['cs'] = v
                            elif k == 'OverallDifficulty':
                                meta['od'] = v
                            elif k == 'ApproachRate':
                                meta['ar'] = v
                            elif k == 'SliderMultiplier':
                                meta['sm'] = v
                            elif k == 'SliderTickRate':
                                meta['str'] = v
                        except ValueError:
                            continue
        except Exception:
            pass
        return meta

    @staticmethod
    def parse(filepath):
        slider_multiplier = 1.0
        timing_points = []
        hit_objects = []
        
        if not os.path.exists(filepath):
            return timing_points, hit_objects, slider_multiplier
            
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
                
        # Parse Difficulty
        if 'Difficulty' in sections:
            for line in sections['Difficulty']:
                if ':' in line:
                    key, val = line.split(':', 1)
                    if key.strip() == 'SliderMultiplier':
                        slider_multiplier = float(val.strip())
                        
        # Parse Timing Points
        if 'TimingPoints' in sections:
            for line in sections['TimingPoints']:
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                try:
                    t = float(parts[0])
                    beat_len = float(parts[1])
                    is_bpm = beat_len > 0
                    
                    timing_points.append({
                        'time': t,
                        'beat_len': beat_len,
                        'is_bpm': is_bpm,
                        'meter': int(parts[2]) if len(parts) > 2 else 4
                    })
                except ValueError:
                    continue
                
        # Sort timing points
        timing_points.sort(key=lambda x: x['time'])
        
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
                        hit_objects.append({
                            'type': 'circle',
                            'start_time': time_ms,
                            'end_time': time_ms
                        })
                    elif is_slider:
                        slides = int(parts[6]) if len(parts) > 6 else 1
                        length = float(parts[7]) if len(parts) > 7 else 0.0
                        
                        # Find active timing point parameters
                        parent_beat_len = 600.0  # Default fallback (100 BPM)
                        sv_multiplier = 1.0
                        
                        for tp in timing_points:
                            if tp['time'] <= time_ms:
                                if tp['is_bpm']:
                                    parent_beat_len = tp['beat_len']
                                    sv_multiplier = 1.0
                                else:
                                    sv_multiplier = -100.0 / tp['beat_len']
                            else:
                                break
                                
                        effective_sv = slider_multiplier * sv_multiplier
                        duration = (length / (100.0 * effective_sv)) * parent_beat_len * slides
                        
                        hit_objects.append({
                            'type': 'slider',
                            'start_time': time_ms,
                            'end_time': time_ms + duration
                        })
                    elif is_spinner:
                        end_time = float(parts[5]) if len(parts) > 5 else time_ms
                        hit_objects.append({
                            'type': 'spinner',
                            'start_time': time_ms,
                            'end_time': end_time
                        })
                except (ValueError, IndexError):
                    continue
                    
        return timing_points, hit_objects, slider_multiplier


def assign_pattern_labels(hit_objects, timing_points):
    """
    Groups consecutive circle hit objects into streams, triplets, or jumps
    based on the time gap between them and the current BPM.
    
    Classes assigned to circle.pattern_class:
      1: Circle O (Jump / Single)
      5: B-Stream
      6: I-Stream
      7: B-Triplet
      8: I-Triplet
    """
    if not hit_objects:
        return hit_objects

    def get_beat_len(t_ms):
        parent_beat_len = 600.0  # Default fallback (100 BPM)
        for tp in timing_points:
            if tp['time'] <= t_ms:
                if tp['is_bpm']:
                    parent_beat_len = tp['beat_len']
            else:
                break
        return parent_beat_len

    # Identify indices of circles in hit_objects
    circle_indices = [idx for idx, ho in enumerate(hit_objects) if ho['type'] == 'circle']
    
    if not circle_indices:
        return hit_objects
        
    # Group consecutive circles by timing gap and index adjacency
    bursts = []
    current_burst = [circle_indices[0]]
    
    for idx in circle_indices[1:]:
        prev_idx = current_burst[-1]
        
        # Break burst if they are not consecutive in the original hit_objects list
        if idx != prev_idx + 1:
            bursts.append(current_burst)
            current_burst = [idx]
            continue
            
        t_prev = hit_objects[prev_idx]['start_time']
        t_curr = hit_objects[idx]['start_time']
        dt = t_curr - t_prev
        
        # Threshold: beat_len / 4 + 15ms
        beat_len = get_beat_len(t_prev)
        threshold = (beat_len / 4.0) + 15.0
        
        if dt <= threshold:
            current_burst.append(idx)
        else:
            bursts.append(current_burst)
            current_burst = [idx]
            
    if current_burst:
        bursts.append(current_burst)
        
    # Assign labels based on burst length
    for burst in bursts:
        L = len(burst)
        if L < 3:
            # Jump / Single
            for idx in burst:
                hit_objects[idx]['pattern_class'] = 1
        elif L == 3:
            # Triplet
            hit_objects[burst[0]]['pattern_class'] = 7  # B-Triplet
            hit_objects[burst[1]]['pattern_class'] = 8  # I-Triplet
            hit_objects[burst[2]]['pattern_class'] = 8  # I-Triplet
        else:
            # Stream
            hit_objects[burst[0]]['pattern_class'] = 5  # B-Stream
            for idx in burst[1:]:
                hit_objects[idx]['pattern_class'] = 6  # I-Stream
                
    return hit_objects

class OsuRhythmDataset(Dataset):
    """Custom Dataset aligning spectrograms and Osu! beatmaps."""
    def __init__(self, maps_dir, spec_dir, sr=11025, hop_length=512, chunk_size=512, onset_width=3, is_train=True, num_classes=9, mapset_ids=None):
        self.maps_dir = maps_dir
        self.spec_dir = spec_dir
        self.sr = sr
        self.hop_length = hop_length
        self.chunk_size = chunk_size
        self.onset_width = onset_width
        self.is_train = is_train
        self.num_classes = num_classes
        self.mapset_ids = mapset_ids
        self.cache = {}
        
        self.valid_pairs = []
        self._find_valid_pairs()
        
    def _find_valid_pairs(self):
        if not os.path.exists(self.maps_dir):
            print(f"Maps directory '{self.maps_dir}' does not exist.")
            return
            
        map_files = [f for f in os.listdir(self.maps_dir) if f.endswith('.osu')]
        print(f"Scanning {len(map_files)} map files for matching spectrograms...")
        
        for map_file in map_files:
            # Map name structure: <mapset_id>_<difficulty>.osu
            # Spectrogram name: <mapset_id>.npy
            mapset_id = map_file.split('_')[0]
            spec_file = f"{mapset_id}.npy"
            spec_path = os.path.join(self.spec_dir, spec_file)
            
            if os.path.exists(spec_path):
                if self.mapset_ids is not None and mapset_id not in self.mapset_ids:
                    continue
                map_path = os.path.join(self.maps_dir, map_file)
                self.valid_pairs.append({
                    'map_path': map_path,
                    'spec_path': spec_path,
                    'mapset_id': mapset_id,
                    'filename': map_file
                })
                
        print(f"Found {len(self.valid_pairs)} valid aligned map-spectrogram pairs.")

    def __len__(self):
        return len(self.valid_pairs)

    def _generate_features_and_labels(self, pair):
        # Load spectrogram: shape (n_bins, n_frames)
        S = np.load(pair['spec_path'])
        if S.shape[1] < 9:
            S = np.pad(S, ((0, 0), (0, 9 - S.shape[1])), mode='edge')
        n_bins, n_frames = S.shape
        
        # Parse map
        timing_points, hit_objects, _ = OsuBeatmapParser.parse(pair['map_path'])
        
        # Group circle hit objects into patterns
        hit_objects = assign_pattern_labels(hit_objects, timing_points)
        
        # 1. Create timing/downbeat channel: shape (n_frames,)
        timing_line = np.zeros(n_frames, dtype=np.float32)
        song_duration_ms = (n_frames * self.hop_length / self.sr) * 1000.0
        
        # Generate beats
        for i, tp in enumerate(timing_points):
            if not tp['is_bpm']:
                continue
            
            # Find boundary (either next timing point or end of song)
            end_t = song_duration_ms
            for next_tp in timing_points[i+1:]:
                if next_tp['is_bpm']:
                    end_t = next_tp['time']
                    break
            
            # Place beat ticks
            t = tp['time']
            beat_idx = 0
            meter = tp['meter']
            while t < end_t:
                # Frame index for this beat
                f = int(round(t * self.sr / (self.hop_length * 1000.0)))
                if 0 <= f < n_frames:
                    # 1.0 for downbeats, 0.5 for regular beats
                    timing_line[f] = 1.0 if (beat_idx % meter == 0) else 0.5
                
                t += tp['beat_len']
                beat_idx += 1

        # 2. Create labels: shape (n_frames,)
        labels = np.zeros(n_frames, dtype=np.int64)
        half_w = self.onset_width // 2
        
        for ho in hit_objects:
            # Helper to place label with widening
            def place_label(center_frame, label_val):
                for f_idx in range(center_frame - half_w, center_frame + half_w + 1):
                    if 0 <= f_idx < n_frames:
                        labels[f_idx] = label_val

            # Circle (Class 1 or pattern classes)
            if ho['type'] == 'circle':
                f = int(round(ho['start_time'] * self.sr / (self.hop_length * 1000.0)))
                label_val = ho.get('pattern_class', 1)
                if self.num_classes == 5 and label_val > 4:
                    label_val = 1
                place_label(f, label_val)
            # Slider (Start: 2, End: 3)
            elif ho['type'] == 'slider':
                fs = int(round(ho['start_time'] * self.sr / (self.hop_length * 1000.0)))
                fe = int(round(ho['end_time'] * self.sr / (self.hop_length * 1000.0)))
                place_label(fs, 2)
                place_label(fe, 3)
            # Spinner (Class 4)
            elif ho['type'] == 'spinner':
                f = int(round(ho['start_time'] * self.sr / (self.hop_length * 1000.0)))
                place_label(f, 4)

        # Compute delta of Mel spectrogram (axis=1 is the time axis)
        import librosa
        S_delta = librosa.feature.delta(S, axis=1)
        
        # Parse difficulty metadata
        meta = OsuBeatmapParser.parse_metadata(pair['map_path'])
        
        # Calculate overall density of hitobjects per second
        song_duration_sec = song_duration_ms / 1000.0
        density = len(hit_objects) / song_duration_sec if song_duration_sec > 0.0 else 0.0
        
        # Combine into a metadata vector
        meta_vec = np.array([
            meta['hp'],
            meta['cs'],
            meta['od'],
            meta['ar'],
            meta['sm'],
            meta['str'],
            density
        ], dtype=np.float32)
        
        # Replicate metadata features across all frames
        meta_grid = np.tile(meta_vec, (n_frames, 1))
        
        # Concatenate Mel, Delta Mel, timing line, and difficulty metadata
        features = np.vstack([S, S_delta, timing_line.reshape(1, -1)]) # shape (169, n_frames)
        features = features.T # shape (n_frames, 169)
        features = np.hstack([features, meta_grid]) # shape (n_frames, 176)
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def get_labels(self, idx):
        # Light parsing to compute label distribution
        if idx in self.cache:
            _, labels = self.cache[idx]
        else:
            pair = self.valid_pairs[idx]
            features, labels = self._generate_features_and_labels(pair)
            self.cache[idx] = (features, labels)
        return labels.numpy()

    def __getitem__(self, idx):
        if idx in self.cache:
            features, labels = self.cache[idx]
        else:
            pair = self.valid_pairs[idx]
            features, labels = self._generate_features_and_labels(pair)
            self.cache[idx] = (features, labels)
            
        n_frames = features.shape[0]
        
        # Perform crop to chunk_size (either random crop for training or middle crop for validation)
        if n_frames > self.chunk_size:
            if self.is_train:
                start = np.random.randint(0, n_frames - self.chunk_size)
            else:
                start = (n_frames - self.chunk_size) // 2
            features = features[start : start + self.chunk_size]
            labels = labels[start : start + self.chunk_size]
        else:
            # Pad if too short
            pad_len = self.chunk_size - n_frames
            features = torch.cat([features, torch.zeros(pad_len, features.shape[1])], dim=0)
            labels = torch.cat([labels, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            
        return features, labels

def compute_class_weights(dataset, num_classes=9, max_weight=20.0, method="square_root"):
    """Computes class weights based on labels in the training subset."""
    print(f"Computing class weights using '{method}' method and caching dataset...")
    class_counts = np.zeros(num_classes)
    total_len = len(dataset)
    for idx in range(total_len):
        if idx % 500 == 0:
            print(f"  Progress: {idx}/{total_len} files cached...")
        _, labels = dataset[idx]
        for c in range(num_classes):
            class_counts[c] += torch.sum(labels == c).item()
    print(f"  Progress: {total_len}/{total_len} files cached.")
            
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    
    if method == "square_root":
        # Square-root inverse frequency: grows more slowly than pure inverse
        weights = 1.0 / np.sqrt(class_counts)
        # Normalize so that the minimum weight (None class) is 1.0
        weights = weights / np.min(weights)
    elif method == "inverse":
        # Standard inverse frequency
        total = np.sum(class_counts)
        weights = total / (float(num_classes) * class_counts)
    else:
        # Uniform weights
        weights = np.ones(num_classes)
        
    if max_weight is not None:
        weights = np.minimum(weights, max_weight)
        
    return torch.tensor(weights, dtype=torch.float)

def snap_to_grid(time_ms, timing_points, allowed_subdivisions=[1, 2, 4]):
    """
    Snaps a timestamp in milliseconds to the nearest valid beat subdivision grid tick.
    allowed_subdivisions: list of integers (1 = 1/1 beat, 2 = 1/2 beat, 4 = 1/4 beat).
    """
    if not timing_points:
        return time_ms

    # Find the active uninherited timing point (BPM change) at time_ms
    active_tp = None
    for tp in timing_points:
        if tp['is_bpm']:
            if active_tp is None or tp['time'] <= time_ms:
                active_tp = tp
            else:
                break
                
    if active_tp is None:
        return time_ms
        
    tp_time = active_tp['time']
    beat_len = active_tp['beat_len']
    
    # Calculate difference in beats since timing point start
    beats_since = (time_ms - tp_time) / beat_len
    
    # Find the closest subdivision tick
    best_snapped_time = time_ms
    min_dist = float('inf')
    
    for div in allowed_subdivisions:
        tick_idx = round(beats_since * div)
        snapped_time = tp_time + (tick_idx / div) * beat_len
        dist = abs(time_ms - snapped_time)
        if dist < min_dist:
            min_dist = dist
            best_snapped_time = snapped_time
            
    # Snap threshold: only snap if within 50ms of the grid
    if min_dist <= 50.0:
        return best_snapped_time
    else:
        return time_ms

def collate_fn_eval(batch):
    """Custom collate function for validation to handle variable-length sequences."""
    # Batch size is 1 for full sequence validation/eval
    return batch[0][0].unsqueeze(0), batch[0][1].unsqueeze(0)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_batches = len(dataloader)
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        if batch_idx % 50 == 0:
            print(f"    Batch {batch_idx}/{total_batches}...")
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(features)
        
        # Reshape logits to (Batch * SeqLen, Classes) and labels to (Batch * SeqLen)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device, num_classes=9):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            logits = model(features)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(labels.view(-1).numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    print("\nClassification Report (Sequence Frame-level):")
    if num_classes == 5:
        target_names = ["None", "Circle", "Slider Start", "Slider End", "Spinner"]
        labels = [0, 1, 2, 3, 4]
    else:
        target_names = ["None", "Circle O", "Slider Start", "Slider End", "Spinner", "B-Stream", "I-Stream", "B-Triplet", "I-Triplet"]
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
    print(classification_report(all_targets, all_preds, target_names=target_names, labels=labels, zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    
    # Calculate onset F1 score (excluding Class 0)
    onset_indices = all_targets > 0
    if np.sum(onset_indices) > 0:
        onset_acc = np.mean(all_preds[onset_indices] == all_targets[onset_indices])
        print(f"Onset Class Accuracy: {onset_acc * 100:.2f}%")
    else:
        print("No onset events found in evaluation set.")

def main():
    parser = argparse.ArgumentParser(description="Train rhythm generation temporal transcription model.")
    parser.add_argument("--maps_dir", type=str, default="maps", help="Maps directory path")
    parser.add_argument("--spec_dir", type=str, default="spectrograms", help="Spectrograms directory path")
    parser.add_argument("--model_type", type=str, choices=["cnn-lstm", "transformer", "cnn-transformer"], default="cnn-lstm", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--chunk_size", type=int, default=512, help="Sequence chunk size for training")
    parser.add_argument("--onset_width", type=int, default=3, help="Note onset frame target width")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation dataset split ratio")
    parser.add_argument("--save_path", type=str, default="rhythm_model.pth", help="Model checkpoint save path")
    parser.add_argument("--evaluate", action="store_true", help="Only run evaluation using saved checkpoint")
    parser.add_argument("--predict_npy", type=str, default=None, help="Path to a spectrogram .npy file to transcribe")
    parser.add_argument("--timing_osu", type=str, default=None, help="Optional path to a .osu file to extract timing points for grid snapping")
    parser.add_argument("--max_weight", type=float, default=20.0, help="Maximum class weight cap to avoid rare classes dominating")
    parser.add_argument("--num_classes", type=int, default=9, help="Number of classes (5 for legacy, 9 for patterns)")
    parser.add_argument("--num_bands", type=int, default=3, choices=[1, 2, 3, 4, 6, 7, 12, 14, 21, 28, 42, 84], help="Number of bands to split the 84 frequency bins into for Channel Attention (default: 3)")
    parser.add_argument("--weight_method", type=str, choices=["inverse", "square_root", "none"], default="square_root", help="Method to calculate class weights for balancing class representation (default: square_root)")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing value for CrossEntropyLoss (default: 0.05)")
    parser.add_argument("--cnn_channels", type=int, default=128, help="Number of CNN channels (default: 128)")
    parser.add_argument("--lstm_hidden", type=int, default=128, help="Number of LSTM hidden units (default: 128)")
    parser.add_argument("--hop_length", type=int, default=512, help="Spectrogram hop length (default: 512)")
    parser.add_argument("--use_focal_loss", action="store_true", help="Use Focal Loss instead of CrossEntropyLoss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss (default: 2.0)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Predict mode
    if args.predict_npy:
        if not os.path.exists(args.predict_npy):
            print(f"Spectrogram file '{args.predict_npy}' not found.")
            return
            
        print(f"Loading model checkpoint from '{args.save_path}' with {args.num_classes} classes...")
        model_classes = {
            "cnn-lstm": CNNLSTMRhythmModel,
            "cnn-transformer": CNNTransformerRhythmModel,
            "transformer": TransformerRhythmModel
        }
        model_class = model_classes.get(args.model_type, CNNLSTMRhythmModel)
        try:
            model = load_model_helper(model_class, args.save_path, device, num_classes=args.num_classes, default_num_bands=args.num_bands)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return
            
        model.eval()
        
        # Load spectrogram (n_bins, n_frames)
        S = np.load(args.predict_npy)
        n_bins, n_frames = S.shape
        
        # Load or estimate timing points for grid snapping and timing channel input
        timing_points = []
        if args.timing_osu:
            if os.path.exists(args.timing_osu):
                print(f"Parsing timing points from {args.timing_osu}...")
                timing_points, _, _ = OsuBeatmapParser.parse(args.timing_osu)
            else:
                print(f"Warning: Timing Osu file '{args.timing_osu}' not found. Defaulting to estimated tempo.")
                
        sr = 11025
        hop_length = 512
        
        if not timing_points:
            print("No timing file provided. Estimating tempo and beats directly from spectrogram using librosa...")
            try:
                import librosa.onset
                # Compute onset strength envelope
                onset_env = librosa.onset.onset_strength(S=S, sr=sr, hop_length=hop_length)
                tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
                tempo = float(tempo)
                beat_len = 60000.0 / tempo
                first_beat_ms = beats[0] * hop_length / sr * 1000.0 if len(beats) > 0 else 0.0
                print(f"Estimated Tempo: {tempo:.2f} BPM (Beat length: {beat_len:.2f} ms, First beat offset: {first_beat_ms:.2f} ms)")
                timing_points = [{
                    'time': first_beat_ms,
                    'beat_len': beat_len,
                    'is_bpm': True,
                    'meter': 4
                }]
            except Exception as e:
                print(f"Failed to estimate tempo: {e}. Defaulting to 120 BPM grid.")
                timing_points = [{
                    'time': 0.0,
                    'beat_len': 500.0,
                    'is_bpm': True,
                    'meter': 4
                }]
                
        # Generate timing point tick channel input using correct beats
        timing_line = np.zeros(n_frames, dtype=np.float32)
        song_duration_ms = (n_frames * hop_length / sr) * 1000.0
        
        for i, tp in enumerate(timing_points):
            if not tp['is_bpm']:
                continue
            
            end_t = song_duration_ms
            for next_tp in timing_points[i+1:]:
                if next_tp['is_bpm']:
                    end_t = next_tp['time']
                    break
                    
            t = tp['time']
            beat_idx = 0
            meter = tp['meter']
            while t < end_t:
                f = int(round(t * sr / (hop_length * 1000.0)))
                if 0 <= f < n_frames:
                    timing_line[f] = 1.0 if (beat_idx % meter == 0) else 0.5
                t += tp['beat_len']
                beat_idx += 1
            
        # Format input feature: (n_frames, 169) using Mel, Delta, and timing line
        import librosa
        S_delta = librosa.feature.delta(S, axis=1)
        features = np.vstack([S, S_delta, timing_line.reshape(1, -1)]).T
        
        # Check if the loaded model expects metadata channels (input_dim > 169)
        expected_dim = 169
        if hasattr(model, 'attention') and hasattr(model.attention, 'input_dim'):
            expected_dim = model.attention.input_dim
            
        if expected_dim > 169:
            # Parse difficulty metadata if timing_osu is available, else use default values
            meta = OsuBeatmapParser.parse_metadata(args.timing_osu) if args.timing_osu else {
                'hp': 5.0, 'cs': 4.0, 'od': 5.0, 'ar': 5.0, 'sm': 1.4, 'str': 1.0
            }
            # Density defaults to 0.0 in predict mode if no hit objects parsed
            density = 0.0
            if args.timing_osu:
                try:
                    _, hit_objects, _ = OsuBeatmapParser.parse(args.timing_osu)
                    song_duration_sec = song_duration_ms / 1000.0
                    density = len(hit_objects) / song_duration_sec if song_duration_sec > 0.0 else 0.0
                except Exception:
                    pass
            meta_vec = np.array([
                meta['hp'], meta['cs'], meta['od'], meta['ar'], meta['sm'], meta['str'], density
            ], dtype=np.float32)
            meta_grid = np.tile(meta_vec, (n_frames, 1))
            features = np.hstack([features, meta_grid])
            
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        
        print("Transcribing rhythm...")
        with torch.no_grad():
            logits = model(features_tensor)
            preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
        # Decode and print events with grid snapping
        print("\nPredicted Rhythm Events:")
        print("----------------------------------------------------------------------")
        print(f"{'Frame':<8} | {'Raw Time':<12} | {'Snapped Time':<12} | {'Snapped Diff':<12} | {'Event':<15}")
        print("----------------------------------------------------------------------")
        if args.num_classes == 5:
            classes = ["None", "Circle", "Slider Start", "Slider End", "Spinner"]
        else:
            classes = ["None", "Circle O", "Slider Start", "Slider End", "Spinner", "B-Stream", "I-Stream", "B-Triplet", "I-Triplet"]
        
        # Find continuous regions of predictions to avoid duplicates from onset widening
        last_class = 0
        cooldown = 0
        for f, pred in enumerate(preds):
            if pred > 0:
                if pred != last_class or cooldown == 0:
                    time_ms = f * hop_length / sr * 1000.0
                    print(f"Time: {time_ms:8.2f} ms | Event: {classes[pred]} (Frame: {f})")
                    last_class = pred
                    cooldown = args.onset_width + 1
            else:
                last_class = 0
                
            if cooldown > 0:
                cooldown -= 1
        return
 
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
        print("No training data found. Please run audio_to_spectrogram.py to generate spectrograms first.")
        sys.exit(1)
        
    np.random.seed(42) # set seed for reproducibility
    np.random.shuffle(all_mapset_ids)
    val_count = int(len(all_mapset_ids) * args.val_split)
    val_mapset_ids = set(all_mapset_ids[:val_count])
    train_mapset_ids = set(all_mapset_ids[val_count:])
    
    print(f"Split {len(all_mapset_ids)} unique songs into {len(train_mapset_ids)} train and {len(val_mapset_ids)} validation songs.")
    
    train_dataset = OsuRhythmDataset(
        maps_dir=args.maps_dir,
        spec_dir=args.spec_dir,
        onset_width=args.onset_width,
        is_train=True,
        num_classes=args.num_classes,
        mapset_ids=train_mapset_ids,
        hop_length=args.hop_length
    )
    val_dataset = OsuRhythmDataset(
        maps_dir=args.maps_dir,
        spec_dir=args.spec_dir,
        onset_width=args.onset_width,
        is_train=False,
        num_classes=args.num_classes,
        mapset_ids=val_mapset_ids,
        hop_length=args.hop_length
    )
    
    if len(train_dataset) == 0:
        print("Train dataset is empty!")
        sys.exit(1)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # Batch size same as training since validation sequences are now cropped to chunk_size
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Set up model
    model_classes = {
        "cnn-lstm": CNNLSTMRhythmModel,
        "cnn-transformer": CNNTransformerRhythmModel,
        "transformer": TransformerRhythmModel
    }
    model_class = model_classes.get(args.model_type, CNNLSTMRhythmModel)
    
    if args.evaluate:
        print(f"Loading checkpoint '{args.save_path}' for evaluation...")
        try:
            model = load_model_helper(model_class, args.save_path, device, num_classes=args.num_classes, default_num_bands=args.num_bands)
        except Exception as e:
            sys.exit(1)
        evaluate_model(model, val_loader, device, num_classes=args.num_classes)
        return
    else:
        # Standard initialization for training (either custom num_bands/shape or auto-resuming shape if check-point exists)
        num_bands = args.num_bands
        cnn_channels = args.cnn_channels
        lstm_hidden = args.lstm_hidden
        d_model = args.cnn_channels
        
        if os.path.exists(args.save_path):
            try:
                ckpt = torch.load(args.save_path, map_location='cpu')
                if 'attention.fc_mel.0.weight' in ckpt:
                    num_bands = ckpt['attention.fc_mel.0.weight'].shape[1]
                if 'cnn.0.weight' in ckpt:
                    cnn_channels = ckpt['cnn.0.weight'].shape[0]
                if 'lstm.weight_ih_l0' in ckpt:
                    lstm_hidden = ckpt['lstm.weight_ih_l0'].shape[0] // 4
                if 'input_projection.weight' in ckpt:
                    d_model = ckpt['input_projection.weight'].shape[0]
                print(f"Auto-resuming: detected existing checkpoint. To preserve weights/shape, using: num_bands={num_bands}, cnn_channels={cnn_channels}, lstm_hidden={lstm_hidden}.")
            except Exception:
                pass
                
        # Determine input dimension dynamically from dataset features
        input_dim = train_dataset[0][0].shape[1]
        print(f"Initializing model with input_dim = {input_dim}")
        
        if model_class.__name__ == "CNNLSTMRhythmModel":
            model = model_class(
                input_dim=input_dim, 
                num_classes=args.num_classes, 
                num_bands=num_bands,
                cnn_channels=cnn_channels,
                lstm_hidden=lstm_hidden
            ).to(device)
        elif model_class.__name__ == "CNNTransformerRhythmModel":
            model = model_class(
                input_dim=input_dim, 
                num_classes=args.num_classes, 
                num_bands=num_bands,
                cnn_channels=cnn_channels,
                d_model=d_model
            ).to(device)
        else:
            model = model_class(
                input_dim=input_dim, 
                num_classes=args.num_classes, 
                num_bands=num_bands
            ).to(device)
        
    # Compute class weights to address label imbalance
    print("Computing class weights...")
    try:
        class_weights = compute_class_weights(
            train_dataset, 
            num_classes=args.num_classes, 
            max_weight=args.max_weight, 
            method=args.weight_method
        )
        print(f"Class weights: {class_weights.numpy()}")
        class_weights = class_weights.to(device)
    except Exception as e:
        print(f"Failed to compute class weights: {e}. Using uniform weights.")
        if args.num_classes == 5:
            class_weights = torch.tensor([1.0, 10.0, 10.0, 10.0, 10.0]).to(device)
        else:
            class_weights = torch.tensor([1.0, 10.0, 10.0, 10.0, 10.0, 15.0, 15.0, 15.0, 15.0]).to(device)
        
    if args.use_focal_loss:
        print(f"Using Focal Loss (gamma={args.focal_gamma})")
        criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"\nStarting training on {len(train_dataset)} beatmaps...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Put dataset in training mode for random crops
        train_dataset.is_train = True
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Put dataset in validation mode for full sequences
        train_dataset.is_train = False
        
        # Compute validation loss
        val_loss = 0
        model.eval()
        total_val = len(val_loader)
        print("  Evaluating on validation set...")
        with torch.no_grad():
            for idx, (features, labels) in enumerate(val_loader):
                if idx % 200 == 0:
                    print(f"    Val file {idx}/{total_val}...")
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
        
        # Save best checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, args.save_path)
            
    print(f"\nTraining completed! Best Validation Loss: {best_loss:.4f}")
    print(f"Model saved to '{args.save_path}'")
    
    # Run final evaluation
    print("\nRunning final evaluation on Validation Set...")
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(args.save_path))
    else:
        model.load_state_dict(torch.load(args.save_path))
    evaluate_model(model, val_loader, device, num_classes=args.num_classes)

if __name__ == "__main__":
    main()

