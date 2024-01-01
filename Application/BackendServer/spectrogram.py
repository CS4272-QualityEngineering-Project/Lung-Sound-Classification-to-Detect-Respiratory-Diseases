import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy
import numpy as np
import tensorflow as tf
from flask import jsonify

from Application.BackendServer.classificationModel import do_primary_prediction, do_secondary_prediction


def generate_mel_spec(audio):
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512))
    return mel


def generate_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=128, n_fft=2048, hop_length=512)
    return mfcc


def generate_chroma(audio):
    chroma = librosa.feature.chroma_stft(y=audio, sr=22050, n_chroma=128, n_fft=2048, hop_length=512)
    return chroma


def createSpectrogram(audio):
    print("createSpectrogram called")

    # Load audio file
    y, sr = librosa.load(audio, sr=22500, duration=6)

    # mean = np.mean(y)
    # std = np.std(y)
    # norm_audio = (y - mean) / std

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

    mel = generate_mel_spec(y)
    mfcc_1 = generate_mfcc(y)
    chroma_1 = generate_chroma(y)

    three_chanel = np.stack((mel, mfcc_1, chroma_1), axis=2)
    plt.figure(figsize=(4, 4))
    plt.imshow(three_chanel)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.show()
    plt.savefig("stacked.png")
    plt.close()
    print("stacked.png saved")

    expanded_sample = tf.expand_dims(three_chanel, axis=0)

    print(expanded_sample.shape)

    result = do_primary_prediction(expanded_sample)
    if not result:
        result_data = {'result': False, 'diseases': {}}
        return jsonify(result_data)
    else:
        secondary_result = do_secondary_prediction(expanded_sample)
        for key, value in secondary_result.items():
            if isinstance(value, np.float32):
                secondary_result[key] = float(value)
        return jsonify(secondary_result)
