import librosa
import librosa.display
import numpy as np
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


def expand_dimension(three_chanel):
    expanded_sample = np.array([three_chanel])
    if len(expanded_sample.shape) == 4:
        return expanded_sample
    else:
        return None


def get_severity_level(diseases):
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
    severities = []
    for disease in diseases:
        severities.append(severity[disease])
    return severities


def convert_probabilities_to_float(probabilities):
    floated_probabilities = []
    for i in probabilities:
        if isinstance(i, np.float32):
            floated_probabilities.append(float(i))
    return floated_probabilities


def createSpectrogram(audio):
    # Load audio file
    y, sr = librosa.load(audio, sr=22500, duration=6)

    mel = generate_mel_spec(y)

    mfcc_1 = generate_mfcc(y)

    chroma_1 = generate_chroma(y)

    three_chanel = np.stack((mel, mfcc_1, chroma_1), axis=2)

    expanded_sample = expand_dimension(three_chanel)

    if expanded_sample is None:
        return None

    result = do_primary_prediction(expanded_sample)

    if not result:
        result_data = {'result': False, 'diseases': {}}
        return jsonify(result_data)

    else:
        secondary_result = do_secondary_prediction(expanded_sample)

        severities = get_severity_level(secondary_result['diseases'])

        secondary_result['severities'] = severities

        floated_probabilities = convert_probabilities_to_float(secondary_result['probabilities'])

        secondary_result['probabilities'] = floated_probabilities

        return secondary_result
