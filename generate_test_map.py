#!/usr/bin/env python
"""
Osu! Beatmap Generator - Test Map Generator
------------------------------------------
This script runs the trained rhythm model on a selected mapset's spectrogram
and generates a playable `.osu` beatmap containing the predicted hit object timings
arranged in a circular pattern.
"""

import os
import sys
import argparse
import math
import numpy as np
import torch

# Our imports
from train_rhythm import OsuBeatmapParser
from rhythm_model import CNNLSTMRhythmModel, TransformerRhythmModel, CNNTransformerRhythmModel, load_model_helper
from spatial_model import CNNLSTMSpatialModel, LSTMSpatialModel, TransformerSpatialModel

def predict_coordinates(predicted_events, spec_path, template_osu_path, model_type, model_path, device, hop_length=512):
    """Predicts absolute x, y coordinates for a sequence of rhythm events using a trained spatial model."""
    if not os.path.exists(spec_path):
        print(f"Error: Spectrogram '{spec_path}' not found for spatial prediction.")
        return []
        
    # Load Spectrogram
    S = np.load(spec_path)
    n_bins, n_frames = S.shape
    sr = 11025
    
    # 1. Initialize Model
    # Load model checkpoint state dict to inspect shape and auto-detect input_dim
    input_dim = 425
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if 'cnn.0.weight' in state_dict:
                input_dim = state_dict['cnn.0.weight'].shape[1]
            elif 'lstm.weight_ih_l0' in state_dict:
                input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
            print(f"Auto-detected spatial input_dim = {input_dim} from checkpoint.")
        except Exception as e:
            print(f"Warning: Failed to auto-detect spatial shapes from checkpoint: {e}")
            
    if model_type == "cnn-lstm":
        model = CNNLSTMSpatialModel(input_dim=input_dim).to(device)
    elif model_type == "lstm":
        model = LSTMSpatialModel(input_dim=input_dim).to(device)
    else:
        model = TransformerSpatialModel(input_dim=input_dim).to(device)
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error: Failed to load spatial model: {e}")
        return []
        
    # 2. Build Features
    features = []
    M = len(predicted_events)
    if M == 0:
        return []
        
    # Map event types to classes: 
    # Circle (1) -> 0, Slider Start (2) -> 1, Spinner (4) -> 2, Standalone Slider End / Other (3) -> 3
    # Pattern classes (5, 6, 7, 8) are also circles
    type_map = {1: 0, 2: 1, 4: 2, 3: 3, 5: 0, 6: 0, 7: 0, 8: 0}
    
    for i in range(M):
        t_ms, ev_type = predicted_events[i]
        
        # Timing delta: log(1 + delta_t)
        if i > 0:
            delta_t = t_ms - predicted_events[i-1][0]
        else:
            delta_t = 500.0 # Default
        delta_t_feat = math.log(1.0 + max(0.0, delta_t))
        
        # Note type one-hot
        type_feat = [0.0] * 4
        cls_idx = type_map.get(ev_type, 3)
        type_feat[cls_idx] = 1.0
        
        # Audio context: 5 frames
        f_center = int(round(t_ms * sr / (hop_length * 1000.0)))
        audio_frames = []
        for offset in [-2, -1, 0, 1, 2]:
            f_idx = f_center + offset
            if 0 <= f_idx < n_frames:
                audio_frames.append(S[:, f_idx])
            else:
                audio_frames.append(np.zeros(n_bins))
        audio_feat = np.concatenate(audio_frames)
        
        feat = np.concatenate([[delta_t_feat], type_feat, audio_feat])
        
        if input_dim > 425:
            # Append difficulty metadata and density features
            meta = OsuBeatmapParser.parse_metadata(template_osu_path)
            
            # Calculate template map object density (objects per second)
            try:
                _, hit_objects, _ = OsuBeatmapParser.parse(template_osu_path)
                song_duration_sec = (n_frames * hop_length / sr)
                density = len(hit_objects) / song_duration_sec if song_duration_sec > 0.0 else 0.0
            except Exception:
                density = 0.0
                
            meta_vec = np.array([
                meta['hp'], meta['cs'], meta['od'], meta['ar'], meta['sm'], meta['str'], density
            ], dtype=np.float32)
            feat = np.concatenate([feat, meta_vec])
            
        features.append(feat)
        
    features = np.array(features, dtype=np.float32)
    features_tensor = torch.tensor(features).unsqueeze(0).to(device) # shape (1, M, input_dim)
    
    with torch.no_grad():
        preds = model(features_tensor).squeeze(0).cpu().numpy() # shape (M, 3)
        
    # Reconstruct absolute coordinates
    coords = []
    curr_x, curr_y = 256.0, 192.0
    curr_angle = 0.0
    
    for i in range(M):
        ev_type = predicted_events[i][1]
        sin_dt, cos_dt, raw_dist = preds[i]
        
        # Softplus for distance
        dist = np.where(raw_dist > 20.0, raw_dist, np.log(1.0 + np.exp(np.clip(raw_dist, -80.0, 20.0))))
        
        # Scale and clamp distance for stream/triplet inside notes (6 and 8) to keep streams tight
        if ev_type in [6, 8]:
            dist = np.clip(dist * 0.3, 20.0, 50.0)
        else:
            dist = np.clip(dist, 20.0, 220.0)
        
        # Angle change
        delta_theta = np.arctan2(sin_dt, cos_dt)
        theta = curr_angle + delta_theta
        
        # Next coordinate
        if i == 0:
            x = 256.0
            y = 192.0
        else:
            x = curr_x + dist * math.cos(theta)
            y = curr_y + dist * math.sin(theta)
            
        # Clamp to screen boundary (with safe margin)
        x = np.clip(x, 20.0, 492.0)
        y = np.clip(y, 20.0, 364.0)
        
        coords.append((x, y))
        
        # Update history
        curr_x, curr_y = x, y
        curr_angle = theta
        
    return coords

def apply_structural_continuity(preds, probs_all, timing_points, S, sr, hop_length, similarity_threshold=0.85):
    """
    Detects musically similar measures using a self-similarity matrix of beat-synchronous
    Mel spectrogram features, and copies raw predictions between similar measures to enforce
    rhythmic continuity.
    """
    n_frames = S.shape[1]
    song_duration_ms = (n_frames * hop_length / sr) * 1000.0
    
    # 1. Generate all beat timestamps and identify downbeats (measure starts)
    beat_times = []
    downbeat_indices = []
    
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
            beat_times.append(t)
            if beat_idx % meter == 0:
                downbeat_indices.append(len(beat_times) - 1)
            t += tp['beat_len']
            beat_idx += 1
            
    if not beat_times:
        # Fallback to 120 BPM
        t = 0.0
        beat_idx = 0
        while t < song_duration_ms:
            beat_times.append(t)
            if beat_idx % 4 == 0:
                downbeat_indices.append(len(beat_times) - 1)
            t += 500.0
            beat_idx += 1
            
    # 2. Group beats into measures
    measures = []
    for m_idx in range(len(downbeat_indices)):
        start_beat_idx = downbeat_indices[m_idx]
        if m_idx + 1 < len(downbeat_indices):
            end_beat_idx = downbeat_indices[m_idx + 1]
        else:
            end_beat_idx = len(beat_times)
            
        m_beat_times = beat_times[start_beat_idx:end_beat_idx]
        if len(m_beat_times) == 0:
            continue
            
        start_time = m_beat_times[0]
        if m_idx + 1 < len(downbeat_indices):
            end_time = beat_times[downbeat_indices[m_idx + 1]]
        else:
            end_time = song_duration_ms
            
        measures.append({
            'index': m_idx,
            'start_time': start_time,
            'end_time': end_time,
            'beat_times': m_beat_times
        })
        
    # 3. Compute beat-synchronous transient/onset features (positive delta Mel)
    # This captures rhythmic changes rather than sustained timbre/instrumentation
    import librosa
    S_delta = librosa.feature.delta(S, axis=1)
    S_onset = np.maximum(0, S_delta)
    
    n_bins = S_onset.shape[0]
    n_beats = len(beat_times)
    S_beat = np.zeros((n_bins, n_beats), dtype=np.float32)
    
    for b in range(n_beats):
        t_start = beat_times[b]
        if b + 1 < n_beats:
            t_end = beat_times[b+1]
        else:
            t_end = song_duration_ms
            
        f_start = int(round(t_start * sr / (hop_length * 1000.0)))
        f_end = int(round(t_end * sr / (hop_length * 1000.0)))
        
        f_start = np.clip(f_start, 0, n_frames - 1)
        f_end = np.clip(f_end, f_start + 1, n_frames)
        
        if f_start < f_end:
            S_beat[:, b] = S_onset[:, f_start:f_end].mean(axis=1)
        else:
            S_beat[:, b] = S_onset[:, f_start]
            
    # 4. Form feature vectors for each measure
    measure_vectors = []
    valid_measures = []
    
    for m in measures:
        first_beat_t = m['beat_times'][0]
        last_beat_t = m['beat_times'][-1]
        
        # Exact index lookup
        first_idx = beat_times.index(first_beat_t)
        last_idx = beat_times.index(last_beat_t)
        
        m_beat_indices = range(first_idx, last_idx + 1)
        m_features = S_beat[:, m_beat_indices]
        
        # Standardize to 4 beats for consistency
        target_beats = 4
        if m_features.shape[1] > target_beats:
            m_features = m_features[:, :target_beats]
        elif m_features.shape[1] < target_beats:
            pad_width = target_beats - m_features.shape[1]
            m_features = np.pad(m_features, ((0, 0), (0, pad_width)), mode='edge')
            
        vector = m_features.flatten()
        # Zero-center to compute Pearson correlation coefficient (prevents positive-bias/silence propagation)
        vector = vector - np.mean(vector)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        else:
            vector = np.zeros_like(vector)
            
        measure_vectors.append(vector)
        valid_measures.append(m)
        
    # 5. Compute similarity matrix and copy predictions
    copied_count = 0
    is_copied = [False] * len(valid_measures)
    
    for j in range(len(valid_measures)):
        m_j = valid_measures[j]
        v_j = measure_vectors[j]
        
        best_i = -1
        best_sim = -1.0
        
        # Look back for similar measures
        for i in range(j):
            v_i = measure_vectors[i]
            sim = np.dot(v_i, v_j)
            
            if sim > similarity_threshold and sim > best_sim:
                best_sim = sim
                best_i = i
                
        if best_i != -1:
            m_i = valid_measures[best_i]
            
            f_start_i = int(round(m_i['start_time'] * sr / (hop_length * 1000.0)))
            f_end_i = int(round(m_i['end_time'] * sr / (hop_length * 1000.0)))
            len_i = f_end_i - f_start_i
            
            f_start_j = int(round(m_j['start_time'] * sr / (hop_length * 1000.0)))
            f_end_j = int(round(m_j['end_time'] * sr / (hop_length * 1000.0)))
            len_j = f_end_j - f_start_j
            
            copy_len = min(len_i, len_j)
            if copy_len > 0:
                preds[f_start_j : f_start_j + copy_len] = preds[f_start_i : f_start_i + copy_len]
                probs_all[f_start_j : f_start_j + copy_len] = probs_all[f_start_i : f_start_i + copy_len]
                
                if len_j > len_i:
                    preds[f_start_j + copy_len : f_end_j] = 0
                    probs_all[f_start_j + copy_len : f_end_j] = 0.0
                    probs_all[f_start_j + copy_len : f_end_j, 0] = 1.0
                    
                is_copied[j] = True
                copied_count += 1
                
    print(f"Structural Continuity: Copied rhythm from similar preceding measures in {copied_count} out of {len(valid_measures)} total measures (threshold: {similarity_threshold}).")

def main():
    parser = argparse.ArgumentParser(description="Generate a test beatmap from trained rhythm model predictions.")
    parser.add_argument("--map_id", type=str, default="1001507", help="Mapset ID (e.g. 1001507)")
    parser.add_argument("--difficulty_idx", type=str, default="0", help="Difficulty index of original map to use as template")
    parser.add_argument("--model_type", type=str, default="cnn-lstm", choices=["cnn-lstm", "transformer", "cnn-transformer"], help="Model type")
    parser.add_argument("--save_path", type=str, default="rhythm_model.pth", help="Model checkpoint path")
    parser.add_argument("--maps_dir", type=str, default="maps", help="Maps directory")
    parser.add_argument("--spec_dir", type=str, default="spectrograms", help="Spectrograms directory")
    parser.add_argument("--onset_width", type=int, default=3, help="Onset frame target width")
    parser.add_argument("--min_prob", type=float, default=0.45, help="Minimum probability threshold for event prediction")
    parser.add_argument("--slider_min_prob", type=float, default=None, help="Minimum probability threshold for slider starts/ends (default: same as min_prob)")
    parser.add_argument("--min_spacing", type=float, default=62.5, help="Minimum spacing between notes in milliseconds")
    parser.add_argument("--offset_ms", type=float, default=0.0, help="Constant timing offset shift in milliseconds (e.g. +30 or -30)")
    parser.add_argument("--snap_res", type=str, default="4", choices=["1", "2", "4"], help="Allowed grid snap subdivisions: 1 (1/1), 2 (1/2), 4 (1/4)")
    parser.add_argument("--use_spatial", action="store_true", help="Use trained spatial model to predict spatial positions")
    parser.add_argument("--spatial_model_path", type=str, default="spatial_model_cnn_lstm.pth", help="Spatial model checkpoint path")
    parser.add_argument("--spatial_model_type", type=str, default="cnn-lstm", choices=["cnn-lstm", "lstm", "transformer"], help="Spatial model type")
    parser.add_argument("--num_classes", type=int, default=9, help="Number of classes (5 for legacy, 9 for patterns)")
    parser.add_argument("--num_bands", type=int, default=3, choices=[1, 2, 3, 4, 6, 7, 12, 14, 21, 28, 42, 84], help="Number of bands to split the 84 frequency bins into for Channel Attention (default: 3)")
    parser.add_argument("--use_similarity", action="store_true", help="Use self-similarity matrix of Mel features to copy-paste rhythms of similar measures for structural continuity")
    parser.add_argument("--similarity_threshold", type=float, default=0.55, help="Pearson correlation threshold of onset features for matching measures (default: 0.55)")
    parser.add_argument("--hop_length", type=int, default=512, help="Spectrogram hop length (default: 512)")
    
    args = parser.parse_args()
    slider_min_prob = args.slider_min_prob if args.slider_min_prob is not None else args.min_prob
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Resolve paths
    template_osu_path = os.path.join(args.maps_dir, f"{args.map_id}_{args.difficulty_idx}.osu")
    spec_path = os.path.join(args.spec_dir, f"{args.map_id}.npy")
    output_osu_path = os.path.join(args.maps_dir, f"{args.map_id}_test.osu")
    
    if not os.path.exists(template_osu_path):
        print(f"Error: Original map template '{template_osu_path}' not found.")
        sys.exit(1)
    if not os.path.exists(spec_path):
        print(f"Error: Spectrogram '{spec_path}' not found.")
        sys.exit(1)
        
    print(f"Template map: {template_osu_path}")
    print(f"Spectrogram: {spec_path}")
    
    # 2. Load Model
    print(f"Loading model ({args.model_type}) from checkpoint '{args.save_path}' with {args.num_classes} classes...")
    model_classes = {
        "cnn-lstm": CNNLSTMRhythmModel,
        "cnn-transformer": CNNTransformerRhythmModel,
        "transformer": TransformerRhythmModel
    }
    model_class = model_classes.get(args.model_type, CNNLSTMRhythmModel)
    try:
        model = load_model_helper(model_class, args.save_path, device, num_classes=args.num_classes, default_num_bands=args.num_bands)
        model.eval()
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        sys.exit(1)
        
    # 3. Transcribe Rhythm
    print("Loading spectrogram...")
    S = np.load(spec_path)
    if S.shape[1] < 9:
        S = np.pad(S, ((0, 0), (0, 9 - S.shape[1])), mode='edge')
    n_bins, n_frames = S.shape
    
    # Extract timing points for grid snapping and timing channel
    timing_points, _, slider_multiplier = OsuBeatmapParser.parse(template_osu_path)
    
    sr = 11025
    hop_length = args.hop_length
    
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
        
    # Format input feature: (n_frames, 169) using Mel and Delta Mel
    import librosa
    S_delta = librosa.feature.delta(S, axis=1)
    features = np.vstack([S, S_delta, timing_line.reshape(1, -1)]).T
    
    # Check if loaded model expects metadata channels (input_dim > 169)
    expected_dim = 169
    if hasattr(model, 'attention') and hasattr(model.attention, 'input_dim'):
        expected_dim = model.attention.input_dim
        
    if expected_dim > 169:
        print(f"Model expects input_dim = {expected_dim}. Appending map difficulty metadata and density features...")
        meta = OsuBeatmapParser.parse_metadata(template_osu_path)
        
        # Calculate template map object density (objects per second)
        _, hit_objects, _ = OsuBeatmapParser.parse(template_osu_path)
        song_duration_sec = song_duration_ms / 1000.0
        density = len(hit_objects) / song_duration_sec if song_duration_sec > 0.0 else 0.0
        
        meta_vec = np.array([
            meta['hp'], meta['cs'], meta['od'], meta['ar'], meta['sm'], meta['str'], density
        ], dtype=np.float32)
        meta_grid = np.tile(meta_vec, (n_frames, 1))
        features = np.hstack([features, meta_grid])
    print("Running predictions in chunks...")
    preds = np.zeros(n_frames, dtype=np.int64)
    probs_all = np.zeros((n_frames, args.num_classes), dtype=np.float32)
    chunk_size = 512
    for start_f in range(0, n_frames, chunk_size):
        end_f = min(start_f + chunk_size, n_frames)
        chunk_features = features[start_f:end_f]
        
        # Pad if short
        actual_len = chunk_features.shape[0]
        if actual_len < chunk_size:
            pad_len = chunk_size - actual_len
            chunk_features = np.pad(chunk_features, ((0, pad_len), (0, 0)), mode='constant')
            
        chunk_tensor = torch.tensor(chunk_features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(chunk_tensor)
            chunk_probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            chunk_preds = np.argmax(chunk_probs, axis=-1)
            
        preds[start_f:end_f] = chunk_preds[:actual_len]
        probs_all[start_f:end_f] = chunk_probs[:actual_len]
        
    # Apply structural continuity (rhythm copy-paste) if requested
    if args.use_similarity:
        apply_structural_continuity(preds, probs_all, timing_points, S, sr, hop_length, args.similarity_threshold)
        
    # 4. Decode predictions with softmax filtering, grid snapping, and spacing constraints
    from train_rhythm import snap_to_grid
    
    # Extract candidate starts and ends
    if args.num_classes == 5:
        classes = ["None", "Circle", "Slider Start", "Slider End", "Spinner"]
        start_classes = [1, 2, 4]
    else:
        classes = ["None", "Circle O", "Slider Start", "Slider End", "Spinner", "B-Stream", "I-Stream", "B-Triplet", "I-Triplet"]
        start_classes = [1, 2, 4, 5, 6, 7, 8]
        
    # Log predictions to file
    os.makedirs("logs", exist_ok=True)
    predictions_txt_path = os.path.join("logs", f"{args.map_id}_predictions.txt")
    predictions_csv_path = os.path.join("logs", f"{args.map_id}_predictions.csv")
    
    print(f"Saving predictions log to:\n  - Text (active only): {predictions_txt_path}\n  - CSV (all frames): {predictions_csv_path}")
    
    # Write human-readable active predictions log
    with open(predictions_txt_path, 'w', encoding='utf-8') as f_txt:
        f_txt.write(f"Active Predictions Log for Map ID: {args.map_id}\n")
        f_txt.write(f"Model: {args.model_type} (Checkpoint: {args.save_path})\n")
        f_txt.write(f"Confidence Threshold: {args.min_prob}\n")
        f_txt.write("=" * 100 + "\n")
        f_txt.write(f"{'Frame':<8} | {'Raw Time':<12} | {'Snapped Time':<12} | {'Event Type':<16} | {'Probability':<12} | {'Alternative Classes (Prob >= 0.1)'}\n")
        f_txt.write("-" * 100 + "\n")
        
        for f in range(n_frames):
            pred_class = preds[f]
            if pred_class > 0:
                prob = probs_all[f, pred_class]
                time_ms = f * hop_length / sr * 1000.0
                snapped_t = snap_to_grid(time_ms, timing_points, allowed_subdivisions=[1, 2, 4]) + args.offset_ms
                
                # Alternatives
                alts = []
                for c in range(args.num_classes):
                    if c != pred_class and probs_all[f, c] >= 0.1:
                        alts.append(f"{classes[c]} ({probs_all[f, c]:.2f})")
                alts_str = ", ".join(alts) if alts else "None"
                
                f_txt.write(f"{f:<8} | {time_ms:10.1f} ms | {snapped_t:10.1f} ms | {classes[pred_class]:<16} | {prob:11.4f} | {alts_str}\n")
                
    # Write full CSV log with all frames and class probabilities
    with open(predictions_csv_path, 'w', encoding='utf-8') as f_csv:
        header = ["Frame", "Time_ms", "Snapped_Time_ms", "Predicted_Class", "Class_Name", "Probability"]
        # Add probability columns for all classes
        for cls_name in classes:
            header.append(f"Prob_{cls_name.replace(' ', '_').replace('-', '_')}")
            
        f_csv.write(",".join(header) + "\n")
        
        for f in range(n_frames):
            pred_class = preds[f]
            prob = probs_all[f, pred_class]
            time_ms = f * hop_length / sr * 1000.0
            snapped_t = snap_to_grid(time_ms, timing_points, allowed_subdivisions=[1, 2, 4]) + args.offset_ms
            
            row = [
                str(f),
                f"{time_ms:.1f}",
                f"{snapped_t:.1f}",
                str(pred_class),
                classes[pred_class],
                f"{prob:.4f}"
            ]
            # Add all individual class probabilities
            for c in range(args.num_classes):
                row.append(f"{probs_all[f, c]:.4f}")
                
            f_csv.write(",".join(row) + "\n")

    # 4a. Gather raw candidates above probability threshold, pairing slider starts and ends
    raw_starts = []
    raw_ends = []
    
    all_slider_starts = []
    all_slider_ends = []
    
    for f in range(n_frames):
        pred_class = preds[f]
        if pred_class > 0:
            prob = probs_all[f, pred_class]
            time_ms = f * hop_length / sr * 1000.0
            cand = {
                'frame': f,
                'raw_time': time_ms,
                'class': pred_class,
                'prob': prob
            }
            if pred_class == 2:
                all_slider_starts.append(cand)
            elif pred_class == 3:
                all_slider_ends.append(cand)
            elif pred_class in start_classes:
                if prob >= args.min_prob:
                    raw_starts.append(cand)
                    
    # Pair slider starts and ends. Both must be above threshold to keep them as a slider.
    kept_starts = set()
    kept_ends = set()
    converted_starts_to_circles = set()
    converted_ends_to_circles = set()
    
    # 1. Pair up starts and ends, keeping both only if BOTH are above threshold.
    # Otherwise, convert the confident candidate(s) to circles.
    for i, start_cand in enumerate(all_slider_starts):
        # Find the matching end (first end after this start and before the next start)
        next_start_time = float('inf')
        if i + 1 < len(all_slider_starts):
            next_start_time = all_slider_starts[i+1]['raw_time']
            
        matched_end = None
        for end_cand in all_slider_ends:
            if start_cand['raw_time'] < end_cand['raw_time'] < next_start_time:
                matched_end = end_cand
                break
                
        if matched_end is not None:
            if start_cand['prob'] >= slider_min_prob and matched_end['prob'] >= slider_min_prob:
                kept_starts.add(start_cand['frame'])
                kept_ends.add(matched_end['frame'])
            else:
                if start_cand['prob'] >= slider_min_prob:
                    converted_starts_to_circles.add(start_cand['frame'])
                if matched_end['prob'] >= slider_min_prob:
                    converted_ends_to_circles.add(matched_end['frame'])
        else:
            if start_cand['prob'] >= slider_min_prob:
                converted_starts_to_circles.add(start_cand['frame'])
                
    # 2. Check for any standalone ends above threshold that weren't matched
    for end_cand in all_slider_ends:
        if end_cand['prob'] >= slider_min_prob:
            if end_cand['frame'] not in kept_ends and end_cand['frame'] not in converted_ends_to_circles:
                converted_ends_to_circles.add(end_cand['frame'])
                
    # Add the kept slider starts/ends to the main candidate lists, converting unmatched ones to circles
    for start_cand in all_slider_starts:
        if start_cand['frame'] in kept_starts:
            raw_starts.append(start_cand)
        elif start_cand['frame'] in converted_starts_to_circles:
            circle_cand = start_cand.copy()
            circle_cand['class'] = 1  # Convert to Circle
            raw_starts.append(circle_cand)
            
    for end_cand in all_slider_ends:
        if end_cand['frame'] in kept_ends:
            raw_ends.append(end_cand)
        elif end_cand['frame'] in converted_ends_to_circles:
            circle_cand = end_cand.copy()
            circle_cand['class'] = 1  # Convert to Circle
            raw_starts.append(circle_cand)
                    
    # 4b. Grid snap all candidates and apply timing offset
    snap_res_val = int(args.snap_res)
    if snap_res_val == 4:
        allowed_subdivisions = [1, 2, 4]
    elif snap_res_val == 2:
        allowed_subdivisions = [1, 2]
    else:
        allowed_subdivisions = [1]
        
    for cand in raw_starts + raw_ends:
        cand['time'] = snap_to_grid(cand['raw_time'], timing_points, allowed_subdivisions=allowed_subdivisions) + args.offset_ms
        
    # Sort candidates by snapped time
    raw_starts.sort(key=lambda x: x['time'])
    raw_ends.sort(key=lambda x: x['time'])
    
    # 4c. Filter starts to enforce spacing
    filtered_starts = []
    for cand in raw_starts:
        if len(filtered_starts) > 0:
            last_start = filtered_starts[-1]
            time_diff = cand['time'] - last_start['time']
            
            # If both are stream/triplet classes, use a lower threshold (45ms) to filter duplicates
            # Otherwise use args.min_spacing
            is_stream_triplet = cand['class'] in [5, 6, 7, 8] and last_start['class'] in [5, 6, 7, 8]
            threshold = 45.0 if is_stream_triplet else args.min_spacing
            
            if time_diff < threshold:
                # Keep the one with the higher probability
                if cand['prob'] > last_start['prob']:
                    filtered_starts[-1] = cand
                continue
        filtered_starts.append(cand)
        
    # 4d. Reconstruct final sequence of events, pairing starts and ends
    predicted_events = []
    spinner_ends = {}
    
    i = 0
    n_starts = len(filtered_starts)
    spinner_clear_until = 0.0
    
    while i < n_starts:
        cand = filtered_starts[i]
        t_ms = cand['time']
        ev_class = cand['class']
        
        # Ensure no hit objects are present while a spinner is active
        if t_ms < spinner_clear_until:
            i += 1
            continue
            
        if ev_class in [1, 5, 6, 7, 8]:
            # Circle or stream/triplet note
            predicted_events.append((t_ms, ev_class))
            i += 1
        elif ev_class == 4:
            # Spinner: Determine its duration based on the next start note
            next_note_time = float('inf')
            for k in range(i + 1, n_starts):
                if filtered_starts[k]['time'] > t_ms:
                    next_note_time = filtered_starts[k]['time']
                    break
                    
            duration = 2000.0
            if next_note_time != float('inf'):
                available_gap = next_note_time - t_ms - 150.0 # 150ms gap before next note
                if available_gap < 1000.0:
                    # Fallback to circle if not enough room for a playable spinner
                    predicted_events.append((t_ms, 1))
                    i += 1
                    continue
                else:
                    duration = min(3000.0, available_gap)
            
            end_t = t_ms + duration
            predicted_events.append((t_ms, 4))
            spinner_ends[t_ms] = end_t
            
            # Set the clearance window so no other hit objects start during this spinner
            spinner_clear_until = end_t + 150.0
            i += 1
        elif ev_class == 2:
            # Slider Start: find the next Slider End that is after t_ms
            slider_end_t = None
            for end_cand in raw_ends:
                if end_cand['time'] > t_ms:
                    # Check that this end isn't after the NEXT start object
                    if i + 1 < n_starts and end_cand['time'] >= filtered_starts[i+1]['time']:
                        break
                    slider_end_t = end_cand['time']
                    break
            
            if slider_end_t is not None:
                predicted_events.append((t_ms, 2))
                predicted_events.append((slider_end_t, 3))
                i += 1
            else:
                # Fallback: instead of defaulting to a Circle, find active beat length
                # and create a default-duration slider (1 beat or 1/2 beat) if there is enough space.
                parent_beat_len = 500.0
                for tp in timing_points:
                    if tp['time'] <= t_ms:
                        if tp['is_bpm']:
                            parent_beat_len = tp['beat_len']
                    else:
                        break
                
                next_start_time = filtered_starts[i+1]['time'] if i + 1 < n_starts else float('inf')
                time_to_next = next_start_time - t_ms
                
                if time_to_next >= parent_beat_len:
                    slider_end_t = t_ms + parent_beat_len
                    predicted_events.append((t_ms, 2))
                    predicted_events.append((slider_end_t, 3))
                elif time_to_next >= parent_beat_len / 2.0:
                    slider_end_t = t_ms + (parent_beat_len / 2.0)
                    predicted_events.append((t_ms, 2))
                    predicted_events.append((slider_end_t, 3))
                else:
                    # Fallback to Circle if next note is too close
                    predicted_events.append((t_ms, 1))
                i += 1
        else:
            i += 1
            
    print(f"Detected {len(predicted_events)} rhythm events after grid snapping and confidence/spacing filtering.")
    
    # Predict coordinates using spatial model if requested
    coords = None
    if args.use_spatial:
        print(f"Running spatial note placement using model '{args.spatial_model_path}'...")
        coords = predict_coordinates(
            predicted_events,
            spec_path,
            template_osu_path,
            args.spatial_model_type,
            args.spatial_model_path,
            device,
            hop_length=args.hop_length
        )
        if len(coords) == 0:
            print("Warning: Spatial placement failed, falling back to circular pattern.")
            coords = None
            
    # 5. Build hit objects layout
    center_x = 256
    center_y = 192
    radius = 120
    angle = 0.0
    angle_step = 0.5  # Increment angle for each hit object to form a circle/spiral
    
    hit_objects_lines = []
    note_count = 0
    
    idx = 0
    n_events = len(predicted_events)
    while idx < n_events:
        t_ms, ev_type = predicted_events[idx]
        if ev_type == 0:
            idx += 1
            continue
            
        # Determine if New Combo (every 8 notes)
        note_count += 1
        is_new_combo = (note_count % 8 == 0)
        
        if ev_type in [1, 5, 6, 7, 8]:
            # Circle or Stream/Triplet note
            obj_type = 5 if is_new_combo else 1
            if coords is not None:
                x = int(round(coords[idx][0]))
                y = int(round(coords[idx][1]))
            else:
                x = int(center_x + radius * math.cos(angle))
                y = int(center_y + radius * math.sin(angle))
                step = 0.15 if ev_type in [6, 8] else angle_step
                angle += step
            hit_objects_lines.append(f"{x},{y},{int(round(t_ms))},{obj_type},0,0:0:0:0:")
            idx += 1
            
        elif ev_type == 2:
            # Slider Start: look ahead for Slider End (type 3)
            slider_end_t = None
            j = idx + 1
            while j < n_events:
                next_t, next_type = predicted_events[j]
                if next_type == 3:
                    slider_end_t = next_t
                    break
                elif next_type == 2:
                    # Found another Slider Start before Slider End, abort searching
                    break
                j += 1
                
            if slider_end_t is not None and slider_end_t > t_ms:
                duration = slider_end_t - t_ms
                
                # Find active beat length at t_ms
                parent_beat_len = 500.0
                for tp in timing_points:
                    if tp['time'] <= t_ms:
                        if tp['is_bpm']:
                            parent_beat_len = tp['beat_len']
                    else:
                        break
                        
                # Calculate slider length (in pixels)
                speed = (100.0 * slider_multiplier) / parent_beat_len
                length = duration * speed
                
                # Place starting point x, y
                if coords is not None:
                    x = int(round(coords[idx][0]))
                    y = int(round(coords[idx][1]))
                    
                    # Direction towards the predicted slider end coordinate
                    x2_pred, y2_pred = coords[j]
                    dx = x2_pred - x
                    dy = y2_pred - y
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist > 0:
                        tx = dx / dist
                        ty = dy / dist
                    else:
                        tx, ty = 1.0, 0.0
                else:
                    x = int(center_x + radius * math.cos(angle))
                    y = int(center_y + radius * math.sin(angle))
                    tx = -math.sin(angle)
                    ty = math.cos(angle)
                    angle += angle_step
                
                # Clamp visual slider offset to 150px
                visual_length = min(length, 150.0)
                x2 = int(round(x + tx * visual_length))
                y2 = int(round(y + ty * visual_length))
                
                obj_type = 6 if is_new_combo else 2
                hit_objects_lines.append(f"{x},{y},{int(round(t_ms))},{obj_type},0,L|{x2}:{y2},1,{length}")
                
                # Mark the matched Slider End as processed
                predicted_events[j] = (next_t, 0)
                idx += 1
            else:
                # No Slider End found, fallback to Circle
                obj_type = 5 if is_new_combo else 1
                if coords is not None:
                    x = int(round(coords[idx][0]))
                    y = int(round(coords[idx][1]))
                else:
                    x = int(center_x + radius * math.cos(angle))
                    y = int(center_y + radius * math.sin(angle))
                    angle += angle_step
                hit_objects_lines.append(f"{x},{y},{int(round(t_ms))},{obj_type},0,0:0:0:0:")
                idx += 1
                
        elif ev_type == 3:
            # Standalone Slider End: treat as Circle
            obj_type = 5 if is_new_combo else 1
            if coords is not None:
                x = int(round(coords[idx][0]))
                y = int(round(coords[idx][1]))
            else:
                x = int(center_x + radius * math.cos(angle))
                y = int(center_y + radius * math.sin(angle))
                angle += angle_step
            hit_objects_lines.append(f"{x},{y},{int(round(t_ms))},{obj_type},0,0:0:0:0:")
            idx += 1
            
        elif ev_type == 4:
            # Spinner
            end_t = spinner_ends.get(t_ms, t_ms + 2000.0)
            obj_type = 12 if is_new_combo else 8
            hit_objects_lines.append(f"256,192,{int(round(t_ms))},{obj_type},0,{int(round(end_t))},0:0:0:0:")
            idx += 1
            
        else:
            idx += 1
        
    # 6. Read template file and build output file content
    with open(template_osu_path, 'r', encoding='utf-8') as f:
        template_lines = f.readlines()
        
    output_lines = []
    in_hit_objects = False
    
    for line in template_lines:
        line_strip = line.strip()
        
        # We need to change the Version (Difficulty name) in [Metadata]
        if line_strip.startswith("Version:"):
            output_lines.append("Version:TestRhythm\n")
            continue
            
        # Stop printing original HitObjects when we hit the section header
        if line_strip == "[HitObjects]":
            in_hit_objects = True
            output_lines.append("[HitObjects]\n")
            # Write our generated hit objects
            for ho_line in hit_objects_lines:
                output_lines.append(ho_line + "\n")
            continue
            
        if in_hit_objects:
            # Skip any lines inside the original [HitObjects] section
            continue
            
        output_lines.append(line)
        
    # 7. Write generated file
    with open(output_osu_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
        
    print(f"\nPlayable test map successfully generated at: {output_osu_path}")
    print("Version/Difficulty name: TestRhythm")
    print("You can copy this file into your Osu! Songs folder under the corresponding map set folder.")

if __name__ == "__main__":
    main()
