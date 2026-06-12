# Osu! Beatmap Generator

A machine learning project to generate Osu! beatmaps from audio files.

## Project Structure

- `data_collector.py`: Collects and downloads Osu! beatmaps and their corresponding audio files (.mp3, .ogg) from the API.
- `audio_to_spectrogram.py`: Converts collected audio files into spectrograms (both CQT and Mel) using `librosa`.
- `model.py`: Model architecture definitions and helper functions for training.
- `osu_beatmap_generator.py`: Generates the final beatmap output file using a trained model.
- `config.py`: Central configuration file for directory paths and API credentials.

## Converting Audio to Spectrograms

The `audio_to_spectrogram.py` script reads audio files from the configured audio folder (default is `audio/`) and transforms them into spectrograms.

### Prerequisites

Make sure the conda environment `osu-beatmap-generator` is active.

### Basic Usage

To run the program and process all audio files in the `audio/` directory:

```bash
# Activate the environment (if not already active)
conda activate osu-beatmap-generator

# Run the program
python audio_to_spectrogram.py
```

### Options

The script supports several command-line flags to customize processing:

- `--input_dir PATH`: Directory containing the audio files (default is the path from `config.py` / `audio/`).
- `--output_dir PATH`: Directory where outputs will be saved (default: `spectrograms/`).
- `--type {cqt,mel}`: The type of spectrogram to calculate (default: `mel` to generate log-mel spectrograms).
- `--sr RATE`: The target sampling rate (default: `11025`).
- `--n_bins BINS`: Number of frequency bins (default: `84`).
- `--limit LIMIT`: Limit the number of files to process (useful for testing).

For example, to process only 5 files:

```bash
python audio_to_spectrogram.py --limit 5
```

## Rhythm Generation (Temporal Transcription)

The `train_rhythm.py` script trains a model to predict note onsets and their types (circles, slider starts, slider ends, spinners) from spectrograms aligned with `.osu` map data. It also supports transcribing new songs and snapping predicted notes to a musical beat grid.

### Training the Model

To train the CNN-LSTM network:
```bash
python train_rhythm.py --epochs 10 --batch_size 16
```

To train the Transformer network:
```bash
python train_rhythm.py --model_type transformer --epochs 10 --batch_size 16
```

### Options for Training

- `--model_type {cnn-lstm,transformer}`: Select the model architecture (default: `cnn-lstm`).
- `--epochs N`: Number of training epochs (default: `10`).
- `--batch_size N`: Batch size for training (default: `16`).
- `--lr LR`: Learning rate (default: `0.001`).
- `--chunk_size FRAMES`: Sequence length for batch training crops (default: `512`).
- `--onset_width FRAMES`: Target label width for note onsets to smooth labels (default: `3`).
- `--val_split RATIO`: Fraction of the dataset to reserve for validation (default: `0.2`).
- `--save_path PATH`: Filename to save/load model checkpoint weights (default: `rhythm_model.pth`).

### Rhythm Transcription and Grid Snapping

During prediction, you can transcribe any `.npy` spectrogram file. To align predictions musically, you can snap predicted notes to the beat grid (subdivisions of 1/1, 1/2, and 1/4 beats):

1. **Snap using an existing `.osu` timing file** (highly recommended if you have a beatmap template):
   ```bash
   python train_rhythm.py --predict_npy spectrograms/100348.npy --timing_osu maps/100348_0.osu
   ```

2. **Snap using Auto-Tempo Estimation** (if you don't have a timing file, `librosa` will automatically estimate the BPM and beat offset from the spectrogram):
   ```bash
   python train_rhythm.py --predict_npy spectrograms/100348.npy
   ```

