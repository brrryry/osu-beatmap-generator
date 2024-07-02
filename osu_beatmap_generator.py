#################################
# Description: This script generates a beatmap for a given song.
# The song is processed through the model and the output is a beatmap.
# NOTE: This script should ONLY be run after the model has been trained.
#################################

# Python library imports
import torch
import numpy as np
import os

# Our imports
import config
from model import convert_to_spectrogram
from model import Encoder, Decoder

# Configuration
model_path = config.model_path
audio_path = config.test_audio_path

####################################
# Helper Functions
####################################


def tensor_to_map(beatmap, output_path):
    """
    Converts a tensor to a beatmap.
    @param beatmap: The beatmap tensor.
    @return: The beatmap.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("[Difficulty]\n")
        f.write(f'HPDrainRate:{difficulties[0]}\n')
        f.write(f'CircleSize:{difficulties[1]}\n')
        f.write(f'OverallDifficulty:{difficulties[2]}\n')
        f.write(f'ApproachRate:{difficulties[3]}\n')
        f.write(f'SliderMultiplier:{difficulties[4]}\n')
        f.write(f'SliderTickRate:{difficulties[5]}\n')
        f.write("\n")
        f.write("[HitObjects]\n")
        for i in range(beatmap.shape[0]):
            for j in range(beatmap.shape[1]):
                if not beatmap[i, j] == 0: print(beatmap[i])
        #Write the timing points



def generate_beatmap(song_file, difficulties, output_path):
    """
    Generates a beatmap for a given song.
    @param song_path: The song file name. (PATH NOT INCLUDED)
    @param output_path: The path to save the beatmap.
    """
    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert song to spectrogram
    spectrogram = convert_to_spectrogram(config.test_audio_path + song_file)
    if isinstance(spectrogram, int):
        print(f"ERROR: Could not convert {song_file} to a spectrogram.")
        return
    spectrogram = torch.tensor(spectrogram.T).float().to(device)

    # Format difficulties
    difficulties = torch.tensor(difficulties).float().to(device)

    # Create the encoder and decoder
    encoder = Encoder(0.4).to(device)
    decoder = Decoder(0.4).to(device)

    # Load the model if it exists
    if not os.path.isfile(model_path + "encoder.pth") or not os.path.isfile(model_path + "decoder.pth"):
        print("ERROR: Model encoder/decoder does not exist. Please train the model first.")
        return
    encoder = Encoder(0.4).to(device)
    decoder = Decoder(0.4).to(device)
    encoder.load_state_dict(torch.load(model_path + "encoder.pth"))
    decoder.load_state_dict(torch.load(model_path + "decoder.pth"))

    # Process the spectrogram
    rhythm, rhythm_hc= encoder(spectrogram)
    beatmap, _, _ = decoder(rhythm, rhythm_hc, difficulties)

    # Convert the beatmap to a file
    tensor_to_map(beatmap, output_path)


# Main
if __name__ == "__main__":
    difficulties = [5, 5, 5, 5, 5, 5]
    generate_beatmap("7484.mp3", difficulties, "output.osu")



