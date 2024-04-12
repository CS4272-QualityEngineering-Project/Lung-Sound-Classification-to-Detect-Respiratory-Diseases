import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy
import numpy as np
import tensorflow as tf
from flask import jsonify

from classificationModel import do_primary_prediction, do_secondary_prediction


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
    y, sr = librosa.load(audio, sr=22500, duration=6 )

    mel = generate_mel_spec(y)
    mfcc_1 = generate_mfcc(y)
    chroma_1 = generate_chroma(y)

    three_chanel = np.stack((mel, mfcc_1, chroma_1), axis=2)
    plt.figure(figsize=(4, 4))
    plt.imshow(three_chanel)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.show()
    # plt.savefig("stacked.png")
    plt.close()
    print("stacked.png saved")

    # expanded_sample = tf.expand_dims(three_chanel, axis=0)
    expanded_sample = np.array([three_chanel])

    print("expanded spec shape",expanded_sample.shape)
    print("expanded spec",expanded_sample)

    result = do_primary_prediction(expanded_sample)
    if not result:
        result_data = {'result': False, 'diseases': {}}
        return jsonify(result_data)
    else:
        severity = {
            "Asthma": 1,
            "Bronchiectasis": 1,
            "Bronchiolitis": 2,
            "Bronchitis": 3,
            "COPD": 1,
            "Lung Fibrosis": 2,
            "Pleural Effusion": 3,
            "Pneumonia": 2,
            "URTI": 2
        }
        secondary_result = do_secondary_prediction(expanded_sample)
        severities = []

        for disease in secondary_result['diseases']:
            severities.append(severity[disease])
        secondary_result['severities'] = severities
        floated_probabilities = []
        for i in secondary_result['probabilities']:
            if isinstance(i, np.float32):
                floated_probabilities.append(float(i))
        secondary_result['probabilities'] = floated_probabilities
        print("secondary_result", secondary_result)
        return secondary_result
