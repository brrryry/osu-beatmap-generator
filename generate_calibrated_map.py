#!/usr/bin/env python
import os
import sys
import math
import argparse
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rhythm'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spatial'))

from train_rhythm import OsuBeatmapParser, get_map_metadata_vector, assign_pattern_labels
from rhythm_model import CNNLSTMRhythmModel, TransformerRhythmModel, CNNTransformerRhythmModel, load_model_helper
from generate_test_map import predict_coordinates, apply_structural_continuity

def main():
    parser = argparse.ArgumentParser(description="Calibrated osu! Beatmap Generator")
    parser.add_argument("--map_id", type=str, default="1001507", help="Mapset ID")
    parser.add_argument("--difficulty_idx", type=str, default="0", help="Difficulty index of original map")
    parser.add_argument("--model_type", type=str, default="cnn-transformer", help="Model type")
    parser.add_argument("--save_path", type=str, default="rhythm/models/rhythm_model_cnn_transformer_5_classes_onset_f1.pth", help="Model checkpoint path")
    parser.add_argument("--maps_dir", type=str, default="data/maps", help="Maps directory")
    parser.add_argument("--spec_dir", type=str, default="data/spectrograms_256", help="Spectrograms directory")
    parser.add_argument("--min_spacing", type=float, default=62.5, help="Minimum spacing in ms")
    parser.add_argument("--offset_ms", type=float, default=0.0, help="Timing offset")
    parser.add_argument("--snap_res", type=str, default="4", help="Grid snap subdivisions")
    parser.add_argument("--use_spatial", action="store_true", default=True, help="Use spatial model")
    parser.add_argument("--spatial_model_path", type=str, default="spatial/models/spatial_model_cnn_lstm.pth", help="Spatial model checkpoint path")
    parser.add_argument("--spatial_model_type", type=str, default="cnn-lstm", help="Spatial model type")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--num_bands", type=int, default=3, help="Number of bands")
    parser.add_argument("--hop_length", type=int, default=512, help="Spectrogram hop length")
    parser.add_argument("--difficulty_name", type=str, default="TestCalibrated_6Star", help="Difficulty name")
    parser.add_argument("--target_sr", type=float, default=6.0, help="Target difficulty star rating override")
    
    # Biases defaults
    parser.add_argument("--circle_bias", type=float, default=3.0, help="Initial circle bias")
    parser.add_argument("--stream_bias", type=float, default=2.0, help="Initial stream bias")
    parser.add_argument("--slider_bias", type=float, default=1.0, help="Initial slider bias")
    
    # Calibrated targets
    parser.add_argument("--target_notes", type=int, default=1000, help="Target total number of notes")
    parser.add_argument("--target_circles_ratio", type=float, default=0.5, help="Target fraction of notes that are circles")
    
    # Smoothing parameters
    parser.add_argument("--smoothing_window", type=float, default=2.0, help="Onset smoothing window in seconds (0.0 to disable)")
    parser.add_argument("--smoothing_alpha", type=float, default=0.8, help="Onset smoothing blend alpha (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    template_osu_path = os.path.join(args.maps_dir, f"{args.map_id}_{args.difficulty_idx}.osu")
    spec_path = os.path.join(args.spec_dir, f"{args.map_id}.npy")
    output_osu_path = os.path.join(args.maps_dir, f"{args.map_id}_test_{args.difficulty_name}.osu")
    
    if not os.path.exists(template_osu_path):
        print(f"Error: Template map '{template_osu_path}' not found.")
        sys.exit(1)
    if not os.path.exists(spec_path):
        print(f"Error: Spectrogram '{spec_path}' not found.")
        sys.exit(1)
        
    # Load model
    print(f"Loading rhythm model ({args.model_type}) from checkpoint '{args.save_path}'...")
    model_classes = {
        "cnn-lstm": CNNLSTMRhythmModel,
        "cnn-transformer": CNNTransformerRhythmModel,
        "transformer": TransformerRhythmModel
    }
    model_class = model_classes.get(args.model_type, CNNLSTMRhythmModel)
    model = load_model_helper(model_class, args.save_path, device, num_classes=args.num_classes, default_num_bands=args.num_bands)
    model.eval()
    
    # Load spectrogram
    S = np.load(spec_path)
    if S.shape[1] < 9:
        S = np.pad(S, ((0, 0), (0, 9 - S.shape[1])), mode='edge')
    n_bins, n_frames = S.shape
    
    timing_points, _, slider_multiplier = OsuBeatmapParser.parse(template_osu_path)
    sr = 11025
    hop_length = args.hop_length
    
    # Generate timing channel
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
            
    import librosa
    S_delta = librosa.feature.delta(S, axis=1)
    features = np.vstack([S, S_delta, timing_line.reshape(1, -1)]).T
    
    # Add metadata features
    expected_dim = 169
    if hasattr(model, 'attention') and hasattr(model.attention, 'input_dim'):
        expected_dim = model.attention.input_dim
        
    if expected_dim > 169:
        print("Appending difficulty metadata vector...")
        parsed_tp, parsed_ho, _ = OsuBeatmapParser.parse(template_osu_path)
        parsed_ho = assign_pattern_labels(parsed_ho, parsed_tp)
        meta_vec, estimated_sr, style = get_map_metadata_vector(template_osu_path, parsed_tp, parsed_ho, sr=11025, hop_length=args.hop_length)
        
        # Force 6.0 star rating in metadata
        print(f"Forcing target SR -> {args.target_sr:.2f}")
        estimated_sr = args.target_sr
        # Group 3 is 6.0 - 8.0 stars
        diff_group = 3
        diff_one_hot = np.zeros(5, dtype=np.float32)
        diff_one_hot[diff_group] = 1.0
        meta_vec[:5] = diff_one_hot
        
        # Override style to balanced (1/3 streams, 1/3 jumps, 1/3 tech)
        style_vars = np.array([0.33, 0.33, 0.33], dtype=np.float32)
        meta_vec[5:] = style_vars
        
        meta_grid = np.tile(meta_vec, (n_frames, 1))
        features = np.hstack([features, meta_grid])
        
    # Predict raw model output
    print("Running raw rhythm model predictions in chunks...")
    probs_onset_all = np.zeros(n_frames, dtype=np.float32)
    probs_type_raw = np.zeros((n_frames, args.num_classes - 1), dtype=np.float32) # excluding class 0
    chunk_size = 512
    
    for start_f in range(0, n_frames, chunk_size):
        end_f = min(start_f + chunk_size, n_frames)
        chunk_features = features[start_f:end_f]
        actual_len = chunk_features.shape[0]
        if actual_len < chunk_size:
            pad_len = chunk_size - actual_len
            chunk_features = np.pad(chunk_features, ((0, pad_len), (0, 0)), mode='constant')
            
        chunk_tensor = torch.tensor(chunk_features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_onset, logits_type = model(chunk_tensor)
            probs_onset = torch.softmax(logits_onset, dim=-1).squeeze(0)
            probs_type = torch.softmax(logits_type, dim=-1).squeeze(0) # shape: (chunk_size, num_classes - 1)
            
            probs_onset_all[start_f:end_f] = probs_onset[:actual_len, 1].cpu().numpy()
            probs_type_raw[start_f:end_f] = probs_type[:actual_len].cpu().numpy()
            
    # Smooth onset probabilities if requested
    if args.smoothing_window > 0.0:
        print(f"Applying onset smoothing over a {args.smoothing_window:.1f}s moving window (alpha={args.smoothing_alpha:.2f})...")
        window_size = int(round(args.smoothing_window * sr / hop_length))
        if window_size % 2 == 0:
            window_size += 1
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            probs_onset_smooth = np.convolve(probs_onset_all, kernel, mode='same')
            probs_onset_all = args.smoothing_alpha * probs_onset_smooth + (1.0 - args.smoothing_alpha) * probs_onset_all
            probs_onset_all = np.clip(probs_onset_all, 0.0, 1.0)
            
    # Class to event mapping (5 classes setup)
    class_to_event = {0: 0, 1: 1, 2: 2, 3: 5, 4: 4}
    
    # ------------------- CALIBRATION LOOP -------------------
    print("\nStarting calibration to hit target notes and ratios...")
    
    best_circle_thresh = 0.45
    best_slider_thresh = 0.45
    
    circle_bias = args.circle_bias
    slider_bias = args.slider_bias
    stream_bias = args.stream_bias
    
    snap_res_val = int(args.snap_res)
    allowed_subdivisions = [1, 2, 4] if snap_res_val == 4 else ([1, 2] if snap_res_val == 2 else [1])
    from train_rhythm import snap_to_grid
    
    def decode_simulation(c_thresh, s_thresh, c_bias, s_bias, st_bias):
        # Apply current biases
        # Classes: 1: Circle, 2: Slider Start, 3: Stream, 4: Spinner
        chunk_probs_note = probs_type_raw.copy()
        chunk_probs_note[:, 0] *= c_bias   # Class 1: Circle
        chunk_probs_note[:, 1] *= s_bias   # Class 2: Slider Start
        chunk_probs_note[:, 2] *= st_bias  # Class 3: Stream
        
        # Argmax over note classes (1-based index)
        best_type = np.argmax(chunk_probs_note, axis=-1) + 1
        
        # Map to thresholds
        thresholds = np.where(best_type == 2, s_thresh, c_thresh)
        preds = np.where(probs_onset_all >= thresholds, best_type, 0)
        
        # Gather candidates
        raw_starts = []
        for f in range(n_frames):
            pred_class = preds[f]
            if pred_class > 0:
                event_type = class_to_event.get(pred_class, pred_class)
                cand = {
                    'frame': f,
                    'raw_time': f * hop_length / sr * 1000.0,
                    'class': event_type,
                    'prob': chunk_probs_note[f, pred_class - 1],
                    'onset_prob': probs_onset_all[f]
                }
                raw_starts.append(cand)
                
        for cand in raw_starts:
            cand['time'] = snap_to_grid(cand['raw_time'], timing_points, allowed_subdivisions=allowed_subdivisions) + args.offset_ms
            
        raw_starts.sort(key=lambda x: x['time'])
        
        # Enforce spacing
        filtered_starts = []
        for cand in raw_starts:
            if len(filtered_starts) > 0:
                last_start = filtered_starts[-1]
                time_diff = cand['time'] - last_start['time']
                is_stream_triplet = cand['class'] in [5, 6, 7, 8] and last_start['class'] in [5, 6, 7, 8]
                spacing_limit = 45.0 if is_stream_triplet else args.min_spacing
                if time_diff < spacing_limit:
                    if cand['prob'] > last_start['prob']:
                        filtered_starts[-1] = cand
                    continue
            filtered_starts.append(cand)
            
        # Reconstruct events and pair sliders
        predicted_events = []
        spinner_ends = {}
        i = 0
        n_starts = len(filtered_starts)
        spinner_clear_until = 0.0
        last_slider_end_time = -1.0
        
        while i < n_starts:
            cand = filtered_starts[i]
            t_ms = cand['time']
            ev_class = cand['class']
            
            if t_ms < spinner_clear_until:
                i += 1
                continue
                
            if ev_class in [1, 5, 6, 7, 8]:
                if last_slider_end_time != -1.0 and (t_ms - last_slider_end_time) < 100.0:
                    i += 1
                    continue
                predicted_events.append((t_ms, ev_class))
                i += 1
            elif ev_class == 4:
                next_note_time = float('inf')
                for k in range(i + 1, n_starts):
                    if filtered_starts[k]['time'] > t_ms:
                        next_note_time = filtered_starts[k]['time']
                        break
                duration = 2000.0
                if next_note_time != float('inf'):
                    available_gap = next_note_time - t_ms - 150.0
                    if available_gap < 1000.0:
                        if cand['onset_prob'] >= c_thresh:
                            predicted_events.append((t_ms, 1))
                        i += 1
                        continue
                    else:
                        duration = min(3000.0, available_gap)
                end_t = t_ms + duration
                predicted_events.append((t_ms, 4))
                spinner_ends[t_ms] = end_t
                spinner_clear_until = end_t + 150.0
                i += 1
            elif ev_class == 2:
                parent_beat_len = 500.0
                for tp in timing_points:
                    if tp['time'] <= t_ms:
                        if tp['is_bpm']:
                            parent_beat_len = tp['beat_len']
                    else:
                        break
                next_start_time = filtered_starts[i+1]['time'] if i + 1 < n_starts else float('inf')
                if next_start_time != float('inf'):
                    gap_ms = next_start_time - t_ms
                    gap_beats = gap_ms / parent_beat_len
                    if gap_beats < 0.5:
                        if cand['onset_prob'] >= c_thresh:
                            predicted_events.append((t_ms, 1))
                        slider_end_t = -1.0
                    else:
                        target_beats = gap_beats - 0.5
                        dur_beats = round(target_beats * 2.0) / 2.0
                        dur_beats = max(0.5, dur_beats)
                        if gap_beats >= 1.0:
                            dur_beats = min(dur_beats, gap_beats - 0.5)
                        else:
                            dur_beats = min(dur_beats, gap_beats - 0.25)
                        dur_beats = round(dur_beats * 4.0) / 4.0
                        dur_beats = max(0.5, dur_beats)
                        dur_beats = min(2.0, dur_beats)
                        
                        duration_ms = dur_beats * parent_beat_len
                        slider_end_t = t_ms + duration_ms
                        predicted_events.append((t_ms, 2))
                        predicted_events.append((slider_end_t, 3))
                else:
                    dur_beats = 1.0
                    duration_ms = dur_beats * parent_beat_len
                    slider_end_t = t_ms + duration_ms
                    predicted_events.append((t_ms, 2))
                    predicted_events.append((slider_end_t, 3))
                last_slider_end_time = slider_end_t
                i += 1
            else:
                i += 1
                
        # Count hit objects
        num_circles = sum(1 for (t, ev) in predicted_events if ev in [1, 4, 5, 6, 7, 8])
        num_sliders = sum(1 for (t, ev) in predicted_events if ev == 2)
        
        return num_circles, num_sliders, predicted_events, spinner_ends
        
    # Outer loop to calibrate biases
    best_loss = float('inf')
    saved_best_state = None
    
    for bias_iter in range(8):
        print(f"Bias Iteration {bias_iter+1}/8 (Circle bias: {circle_bias:.2f}, Slider bias: {slider_bias:.2f}, Stream bias: {stream_bias:.2f})")
        
        # Coordinate descent over thresholds
        c_low, c_high = 0.01, 0.99
        s_low, s_high = 0.01, 0.99
        
        for cd_iter in range(6):
            # 1. Search circle threshold
            for _ in range(8):
                c_mid = (c_low + c_high) / 2.0
                nc, ns, _, _ = decode_simulation(c_mid, (s_low + s_high) / 2.0, circle_bias, slider_bias, stream_bias)
                if nc > args.target_notes * args.target_circles_ratio:
                    c_low = c_mid  # Needs higher threshold to reduce notes
                else:
                    c_high = c_mid
                    
            # 2. Search slider threshold
            for _ in range(8):
                s_mid = (s_low + s_high) / 2.0
                nc, ns, _, _ = decode_simulation((c_low + c_high) / 2.0, s_mid, circle_bias, slider_bias, stream_bias)
                if ns > args.target_notes * (1.0 - args.target_circles_ratio):
                    s_low = s_mid
                else:
                    s_high = s_mid
                    
        # Evaluate current best
        c_thresh = (c_low + c_high) / 2.0
        s_thresh = (s_low + s_high) / 2.0
        nc, ns, evs, spinner_ends = decode_simulation(c_thresh, s_thresh, circle_bias, slider_bias, stream_bias)
        total = nc + ns
        print(f"  -> Thresholds: Circle={c_thresh:.3f}, Slider={s_thresh:.3f}. Result: Circles={nc}, Sliders={ns}, Total={total}")
        
        # Save best state seen so far
        loss = (total - args.target_notes) ** 2 + 4.0 * (nc - ns) ** 2
        if loss < best_loss:
            best_loss = loss
            saved_best_state = {
                'c_thresh': c_thresh,
                's_thresh': s_thresh,
                'c_bias': circle_bias,
                's_bias': slider_bias,
                'st_bias': stream_bias,
                'nc': nc,
                'ns': ns,
                'events': evs,
                'spinner_ends': spinner_ends
            }
            
        # Check if we are close enough (within 5% of target and balanced)
        if abs(total - args.target_notes) < 50 and abs(nc - ns) < 60:
            print("  -> Calibration targets met successfully!")
            break
            
        # If not close enough, adjust biases to shift candidate availability
        # Check if sliders are too low even at lowest threshold
        _, max_sliders, _, _ = decode_simulation(c_thresh, 0.01, circle_bias, slider_bias, stream_bias)
        _, max_circles, _, _ = decode_simulation(0.01, s_thresh, circle_bias, slider_bias, stream_bias)
        
        if max_sliders < args.target_notes * (1.0 - args.target_circles_ratio):
            print(f"  -> WARNING: Slider count is capped at {max_sliders}. Boosting slider bias.")
            slider_bias *= 1.4
            circle_bias *= 0.85
            stream_bias *= 0.85
        elif max_circles < args.target_notes * args.target_circles_ratio:
            print(f"  -> WARNING: Circle count is capped at {max_circles}. Boosting circle bias.")
            circle_bias *= 1.4
            stream_bias *= 1.4
            slider_bias *= 0.85
        else:
            # Adjust biases based on current imbalance
            if nc > ns:
                slider_bias *= 1.15
                circle_bias *= 0.9
                stream_bias *= 0.9
            else:
                circle_bias *= 1.15
                stream_bias *= 1.15
                slider_bias *= 0.9
                
        # Clamp biases to a reasonable range to prevent numerical explosion/collapse
        circle_bias = min(4.0, max(0.5, circle_bias))
        slider_bias = min(4.0, max(0.5, slider_bias))
        stream_bias = min(4.0, max(0.5, stream_bias))
        
        best_circle_thresh = c_thresh
        best_slider_thresh = s_thresh

    # Restore the best calibrated state
    if saved_best_state is not None:
        best_circle_thresh = saved_best_state['c_thresh']
        best_slider_thresh = saved_best_state['s_thresh']
        circle_bias = saved_best_state['c_bias']
        slider_bias = saved_best_state['s_bias']
        stream_bias = saved_best_state['st_bias']
        nc = saved_best_state['nc']
        ns = saved_best_state['ns']
        final_events = saved_best_state['events']
        spinner_ends = saved_best_state['spinner_ends']
        print(f"Restored best state with loss {best_loss:.1f} (Circles: {nc}, Sliders: {ns})")
    else:
        nc, ns, final_events, spinner_ends = decode_simulation(best_circle_thresh, best_slider_thresh, circle_bias, slider_bias, stream_bias)
    print(f"\nFinal Calibrated Output: {nc} Circles, {ns} Sliders. Total: {nc+ns} hit objects.")
    
    # Coordinate Prediction
    coords = None
    if args.use_spatial:
        print(f"Running spatial note placement using model '{args.spatial_model_path}'...")
        coords = predict_coordinates(
            final_events,
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
            
    # Build hit objects layout
    center_x = 256
    center_y = 192
    radius = 120
    angle = 0.0
    angle_step = 0.5
    
    hit_objects_lines = []
    note_count = 0
    idx = 0
    n_events = len(final_events)
    
    while idx < n_events:
        t_ms, ev_type = final_events[idx]
        if ev_type == 0:
            idx += 1
            continue
            
        note_count += 1
        is_new_combo = (note_count % 8 == 0)
        
        if ev_type in [1, 5, 6, 7, 8]:
            obj_type = 5 if is_new_combo else 1
            if coords is not None:
                x = int(round(coords[idx][0]))
                y = int(round(coords[idx][1]))
            else:
                x = int(center_x + radius * math.cos(angle))
                y = int(center_y + radius * math.sin(angle))
                is_inside_stream = ev_type in [6, 7, 8] or (ev_type == 5 and idx > 0 and final_events[idx-1][1] == 5)
                step = 0.15 if is_inside_stream else angle_step
                angle += step
            hit_objects_lines.append(f"{x},{y},{int(round(t_ms))},{obj_type},0,0:0:0:0:")
            idx += 1
            
        elif ev_type == 2:
            slider_end_t = None
            j = idx + 1
            while j < n_events:
                next_t, next_type = final_events[j]
                if next_type == 3:
                    slider_end_t = next_t
                    break
                elif next_type == 2:
                    break
                j += 1
                
            if slider_end_t is not None and slider_end_t > t_ms:
                duration = slider_end_t - t_ms
                parent_beat_len = 500.0
                for tp in timing_points:
                    if tp['time'] <= t_ms:
                        if tp['is_bpm']:
                            parent_beat_len = tp['beat_len']
                    else:
                        break
                speed = (100.0 * slider_multiplier) / parent_beat_len
                length = duration * speed
                
                if coords is not None:
                    x = int(round(coords[idx][0]))
                    y = int(round(coords[idx][1]))
                    x2_pred, y2_pred = coords[j]
                    dx = x2_pred - x
                    dy = y2_pred - y
                    dist = math.sqrt(dx*dx + dy*dy)
                    tx, ty = (dx / dist, dy / dist) if dist > 0 else (1.0, 0.0)
                else:
                    x = int(center_x + radius * math.cos(angle))
                    y = int(center_y + radius * math.sin(angle))
                    tx = -math.sin(angle)
                    ty = math.cos(angle)
                    angle += angle_step
                    
                visual_length = min(length, 150.0)
                x2 = int(round(x + tx * visual_length))
                y2 = int(round(y + ty * visual_length))
                x2_clamped = np.clip(x2, 20.0, 492.0)
                y2_clamped = np.clip(y2, 20.0, 364.0)
                dx_clamped = x2_clamped - x
                dy_clamped = y2_clamped - y
                dist_clamped = math.sqrt(dx_clamped*dx_clamped + dy_clamped*dy_clamped)
                
                if dist_clamped < 30.0:
                    tx, ty = -tx, -ty
                    x2 = int(round(x + tx * visual_length))
                    y2 = int(round(y + ty * visual_length))
                    x2_clamped = np.clip(x2, 20.0, 492.0)
                    y2_clamped = np.clip(y2, 20.0, 364.0)
                    dist_clamped = math.sqrt((x2_clamped-x)**2 + (y2_clamped-y)**2)
                    
                x2, y2 = int(round(x2_clamped)), int(round(y2_clamped))
                visual_length = max(10.0, dist_clamped)
                obj_type = 6 if is_new_combo else 2
                hit_objects_lines.append(f"{x},{y},{int(round(t_ms))},{obj_type},0,L|{x2}:{y2},1,{visual_length}")
                final_events[j] = (next_t, 0)
                idx += 1
            else:
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
            end_t = spinner_ends.get(t_ms, t_ms + 2000.0)
            obj_type = 12 if is_new_combo else 8
            hit_objects_lines.append(f"256,192,{int(round(t_ms))},{obj_type},0,{int(round(end_t))},0:0:0:0:")
            idx += 1
        else:
            idx += 1
            
    # Write to file
    with open(template_osu_path, 'r', encoding='utf-8') as f:
        template_lines = f.readlines()
        
    output_lines = []
    in_hit_objects = False
    
    for line in template_lines:
        line_strip = line.strip()
        if line_strip.startswith("Version:"):
            output_lines.append(f"Version:{args.difficulty_name}\n")
            continue
        if line_strip == "[HitObjects]":
            in_hit_objects = True
            output_lines.append("[HitObjects]\n")
            for ho_line in hit_objects_lines:
                output_lines.append(ho_line + "\n")
            continue
        if in_hit_objects:
            continue
        output_lines.append(line)
        
    with open(output_osu_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
        
    print(f"\nSuccessfully generated calibrated map at: {output_osu_path}")
    print(f"Final Count of Circles/Streams/Spinners: {nc}")
    print(f"Final Count of Sliders: {ns}")
    print(f"Final Star Rating Target: {args.target_sr}")
    print(f"Difficulty/Version Name: {args.difficulty_name}")

if __name__ == "__main__":
    main()
