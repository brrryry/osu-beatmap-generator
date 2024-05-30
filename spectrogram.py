import librosa
from matplotlib import pyplot as plt
import numpy as np

def convert_to_spectrogram(filename):
    targetsamplerate = 11025
    # Load the audio file
    y, sr = librosa.load(filename, sr=targetsamplerate)
    C = np.abs(librosa.cqt(y, sr=targetsamplerate))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                               sr=targetsamplerate, x_axis='time', y_axis='cqt_note', ax=ax)
    ax.set_title('Constant-Q power spectrum')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()
    print("Yippee")

convert_to_spectrogram("test.mp3")