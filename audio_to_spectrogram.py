#!/usr/bin/env python
"""
Audio to Spectrogram Generator
------------------------------
This script processes audio files (MP3, OGG, WAV, etc.) in the audio folder
and converts them into spectrograms, saving the results as NumPy arrays (.npy)
for model training.

It uses librosa's Log-Mel Spectrogram by default, but also supports Constant-Q Transform (CQT).
"""

import os
import sys
import argparse
import time
import traceback
import numpy as np
import librosa
import scipy

# Verify librosa is imported correctly and not shadowed
try:
    _ = librosa.load
except AttributeError as e:
    print(f"Error: {e}", file=sys.stderr)
    print(f"Imported librosa from: {getattr(librosa, '__file__', 'unknown')}", file=sys.stderr)
    print("\nIt seems you are running in a Python environment where 'librosa' is missing attributes,", file=sys.stderr)
    print("shadowed by a local file, or not installed correctly.", file=sys.stderr)
    print("Please make sure you have activated the correct conda environment first:", file=sys.stderr)
    print("    conda activate osu-beatmap-generator", file=sys.stderr)
    sys.exit(1)

# Try importing project config
try:
    import config
    DEFAULT_AUDIO_DIR = config.audio_path
except ImportError:
    DEFAULT_AUDIO_DIR = "audio/"

def parse_args():
    parser = argparse.ArgumentParser(description="Convert collected audio files to spectrogram NumPy arrays.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_AUDIO_DIR,
        help=f"Directory containing audio files (default: {DEFAULT_AUDIO_DIR})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="spectrograms",
        help="Directory to save spectrograms (default: spectrograms/)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["cqt", "mel"],
        default="mel",
        help="Type of spectrogram to generate: 'mel' (Log-Mel Spectrogram) or 'cqt' (Constant-Q Transform) (default: mel)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=11025,
        help="Target sample rate for loading audio (default: 11025)"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=84,
        help="Number of frequency bins (default: 84)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of audio files to process (useful for testing)"
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=512,
        help="Spectrogram hop length (default: 512)"
    )
    return parser.parse_args()

def generate_spectrogram(file_path, sr=11025, spec_type="cqt", n_bins=84, hop_length=512):
    """Loads audio file and computes the spectrogram."""
    # Load audio
    y, actual_sr = librosa.load(file_path, sr=sr)
    
    if spec_type == "cqt":
        # Constant-Q Transform (CQT) divided into 3 bands for Channel Attention:
        # - Low Band: Bins 0-27
        # - Mid Band: Bins 28-55
        # - High Band: Bins 56-83
        C = np.abs(librosa.cqt(y, sr=sr, n_bins=n_bins, bins_per_octave=12, hop_length=hop_length))
        S = librosa.amplitude_to_db(C, ref=np.max)
    elif spec_type == "mel":
        # Log-Mel Spectrogram divided into 3 bands for Channel Attention:
        # - Low Band (Bass): Bins 0-27
        # - Mid Band (Mids): Bins 28-55
        # - High Band (Treble): Bins 56-83
        S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_bins, hop_length=hop_length)
        S = librosa.power_to_db(S_mel, ref=np.max)

        # frequency smoothing
        for i in range(S.shape[0]):
            S[i] = scipy.ndimage.gaussian_filter1d(S[i], sigma=1)

        # multi-band split into 3 bands: low (20-200hz), mid (200-2000hz), high (2000-20000hz)
        low = S[:20]
        mid = S[20:64]
        high = S[64:]
        S = np.concatenate([low, mid, high], axis=0)

    else:
        raise ValueError(f"Unknown spectrogram type: {spec_type}")
        
    return S

def main():
    args = parse_args()
    
    # Normalize paths
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Ensure folder exists
    os.makedirs(output_dir, exist_ok=True)
        
    print(f"=== Audio to Spectrogram Processor ===")
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Spectrogram Type: {args.type.upper()}")
    print(f"Sample Rate:      {args.sr} Hz")
    print(f"Hop Length:       {args.hop_length}")
    print(f"Frequency Bins:   {args.n_bins}")
    print(f"======================================")
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
        
    # Supported audio extensions
    valid_extensions = ('.mp3', '.ogg', '.wav', '.flac', '.m4a')
    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    total_files = len(audio_files)
    if total_files == 0:
        print(f"No valid audio files found in '{input_dir}'.")
        sys.exit(0)
        
    if args.limit:
        audio_files = audio_files[:args.limit]
        print(f"Limiting processing to the first {len(audio_files)} files.")
        total_files = len(audio_files)
        
    print(f"Found {total_files} audio files to process.\n")
    
    success_count = 0
    start_time = time.time()
    
    for idx, audio_file in enumerate(audio_files, 1):
        audio_path = os.path.join(input_dir, audio_file)
        base_name = os.path.splitext(audio_file)[0]
        npy_path = os.path.join(output_dir, f"{base_name}.npy")
        
        if os.path.exists(npy_path) and os.path.getsize(npy_path) > 0:
            print(f"[{idx}/{total_files}] Skipping (already exists): {audio_file}")
            success_count += 1
            continue

        print(f"[{idx}/{total_files}] Processing: {audio_file}...", end="", flush=True)
        file_start = time.time()
        
        try:
            # Generate spectrogram
            S = generate_spectrogram(
                audio_path,
                sr=args.sr,
                spec_type=args.type,
                n_bins=args.n_bins,
                hop_length=args.hop_length
            )
            
            # Save NumPy file (.npy)
            np.save(npy_path, S)
                
            elapsed = time.time() - file_start
            print(f" Done ({elapsed:.2f}s)")
            success_count += 1
            
        except Exception as e:
            print(f" ERROR")
            print(f"Failed to process {audio_file}:", file=sys.stderr)
            traceback.print_exc()
            print("-" * 40, file=sys.stderr)
            
    total_elapsed = time.time() - start_time
    print(f"\n======================================")
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{total_files} files")
    print(f"Total time elapsed:     {total_elapsed:.2f} seconds")
    print(f"======================================")

if __name__ == "__main__":
    main()
