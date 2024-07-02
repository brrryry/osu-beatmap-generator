##########################
# This file contains the classes that make up the model.
##########################


# Python library imports
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from functools import reduce
import gc
import datetime
import traceback


# Our imports
import config

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# num_features
num_features = 8

# Classes
class Audio2Map(torch.utils.data.Dataset):
    """
    A class that represents the dataset of beatmaps.
    """
    def __init__(self, audio_dir, map_dir, pickle_dir):
        """
        Initializes the dataset.
        """
        self.audio_dir = audio_dir
        self.map_dir = map_dir
        self.pickle_dir

    def __len__(self):
        return len([name for name in os.listdir(self.tar_dir) if os.path.isfile(os.path.join(self.tar_dir, name))])

    def __getitem__(self, idx):
        # use the listdir() index 5Head
        # Get the current map name w/o .osu
        files = [name for name in os.listdir(self.tar_dir) if os.path.isfile(os.path.join(self.tar_dir, name))]
        idx -= self.deleted_counter
        idx = min(idx, len(files) - 1)
        print(f"files length: {len(files)}, index: {idx}")
        currfile = files[idx][:-4]
        spec = convert_to_spectrogram(os.path.join(self.in_dir, currfile.split('_', 1)[0] + ".mp3"))
        while(type(spec) == type(None)):
            idx += 1
            currfile = files[idx][:-4]
            spec = convert_to_spectrogram(os.path.join(self.in_dir, currfile.split('_', 1)[0] + ".mp3"))
            if isinstance(spec, int):
                print(f'Could not get item at index {idx} due to parsing spectrogram.') 
                return -1
        input = torch.tensor(spec.T).float()
        diff = parse_difficulty(os.path.join(self.maps_dir, currfile + ".osu"))
        if isinstance(diff, int):
            print(f'Could not get item at index {idx} due to parsing difficulty.') 
            return -1
        diff = torch.t(parse_difficulty(os.path.join(self.maps_dir, currfile + ".osu"))).float()
        out = get_pkl(os.path.join(self.tar_dir, currfile + ".pkl"))[0].to_dense().float()
        if isinstance(out, int):
            print(f'Could not get item at index {idx} due to parsing pkl.') 
            return -1
        return input, diff, out


class Encoder(torch.nn.Module):
    """
    A class that represents the encoder of the model.
    This class will take in a spectrogram and output a list of times.
    """
    def __init__(self, dropout=0.2):
        super(Encoder, self).__init__()
        self.audio_dim = 84
        self.hidden_dim = 64
        self.lstm = torch.nn.LSTM(self.audio_dim, self.hidden_dim, batch_first=True, bidirectional=True, device=device)
        self.dropout = torch.nn.Dropout(dropout)
        #Arbitrary numbers, may be changed later for parameter optimization
    def forward(self, x):
        out, hidden = self.lstm(x)
        out = self.dropout(out)
        return out, hidden

class Decoder(torch.nn.Module):
    """
    A class that represents the decoder of the model.
    """
    def __init__(self, dropout=0.2):
        super(Decoder, self).__init__()
        self.hidden_dim = 64
        self.lstm = torch.nn.LSTM(num_features + 6, self.hidden_dim, num_layers=2, batch_first=True, device=device)
        self.dropout = torch.nn.Dropout(dropout)
        self.hiddenfc = torch.nn.Linear(self.hidden_dim, self.hidden_dim//2, device=device)
        self.outputfc = torch.nn.Linear(self.hidden_dim//2, num_features, device=device)

    def forward(self, encoder_out, encoder_hc, difficulty, target=None):
        decoder_input = torch.zeros((1, num_features + 6), device=device)
        decoder_hidden = encoder_hc
        decoder_outputs = []

        prev_percent = 0
        #currStop = torch.cat((STOP.to(device), difficulty.unsqueeze(0)), 1)

        #while(not torch.equal(decoder_input, currStop)):
        iter = target.shape[0] if target != None else librosa.get_duration(S=encoder_out.T, sr=11025)*100
        # (num samples/sr)*1000 = time in ms
        for i in range(int(iter)):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_output = torch.round(decoder_output)
            decoder_outputs.append(decoder_output.detach())

            if target is not None:
                curr_percent = floor(((i+1)/target.shape[0])*100)
                if curr_percent > prev_percent:
                    prev_percent = curr_percent
                    #print(f"Training...{curr_percent}%")
                decoder_input = torch.cat((target[i], difficulty), 0).unsqueeze(0)
            else:
                if (i+1) % 1000 == 0:
                    print(decoder_output)
                decoder_input = torch.cat((decoder_output, difficulty.unsqueeze(0)), 1).detach()

        decoder_outputs = torch.cat(decoder_outputs, 0)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, x, hc):
        x, hc = self.lstm(x, hc)
        drp = self.dropout(x)
        hidden = self.hiddenfc(drp)
        out = self.outputfc(hidden)
        return out, hc

def tsprint(s):
    """
    Prints a string with a timestamp in front of it.
    """
    print("[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] " + s)



def convert_to_spectrogram(filename):
    try:
        targetSampleRate = 11025
        y, sr = librosa.load(filename, sr=targetSampleRate)
        C = np.abs(librosa.cqt(y, sr=targetSampleRate, n_bins=84, bins_per_octave=12))
        S = librosa.amplitude_to_db(C, ref=np.max)
        #plot the spectrogram

        '''plt.figure(figsize=(12, 4))
        librosa.display.specshow(S, sr=targetSampleRate, x_axis='time', y_axis='cqt_note')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q power spectrogram')
        plt.tight_layout()
        plt.show()'''
        return S
    except:
        tsprint("ERROR: cannot convert " + filename + " to spectrogram.")
        traceback.print_exc()

def print_details():
    """
    Print the details and stats of the model that is about to be trained.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("------------------------------------")
    print("STATS:")
    print("Using device: " + str(device))
    print("Model path: " + config.model_path)
    print("Log path: " + config.log_path)
    print("Number of maps: " + str(get_file_count(config.pickle_path)))
    print("Number of audio files: " + str(get_file_count(config.audio_path)))
    print("------------------------------------")


if __name__ == "__main__":
    print("This file is still in testing. Please do NOT run this file as a script.")
    