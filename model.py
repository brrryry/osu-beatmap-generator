from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import numpy as np
import librosa



def convert_to_spectrogram(filename):
    # Load the audio file
    y, sr = librosa.load(filename)
    C = np.abs(librosa.cqt(y, sr=sr))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                                sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
    ax.set_title('Constant-Q power spectrum')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    print(C)
    plt.show()
    np.savetxt("test.txt", C)
    
   



if __name__ == "__main__":
    convert_to_spectrogram("test.mp3")