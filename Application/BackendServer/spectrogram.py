
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def createSpectrogram(audio):
    print("createSpectrogram called")

    # Load audio file
    y, sr = librosa.load(audio, sr=22500, duration=6)

    # Extract spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

    # Plot Mel spectrogram
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', x_axis='time')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.show()
    plt.savefig("mel_spec_plot.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    print("mel_spec_plot.png saved")

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, n_mfcc=128, n_fft=2048, hop_length=512)
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.show()
    plt.savefig("mfcc_plot.png")
    plt.close()
    print("mfcc_plot.png saved")

    # Extract Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=22050, n_chroma=128, n_fft=2048, hop_length=512)
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.show()
    plt.savefig("chroma.png")
    plt.close()
    print("chroma.png saved")

    three_chanel = np.stack((spectrogram, mfcc, chroma), axis=2)
    plt.figure(figsize=(3, 3))
    plt.imshow(three_chanel)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.show()
    plt.savefig("stacked.png")
    plt.close()
    print("stacked.png saved")
