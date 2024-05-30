import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

def convert_to_spectrogram(filename):
    targetSampleRate = 10000
    y, sr = librosa.load(filename, sr=targetSampleRate, res_type='soxr_lq')
    C = np.abs(librosa.cqt(y, sr=targetSampleRate))
    S = librosa.amplitude_to_db(C, ref=np.max)
    #plot the spectrogram
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                                sr=targetSampleRate, x_axis='time', y_axis='cqt_note', ax=ax)
    ax.set_title('Constant-Q power spectrum')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()
    plt.savefig("spectrogram.png")
    return S

convert_to_spectrogram("test.mp3")