import librosa
import librosa.display
import numpy as np
from flask import jsonify
from matplotlib import pyplot as plt

from classificationModel import do_primary_prediction, do_secondary_prediction


def generate_mel_spec(audio):
    try:
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512))
        return mel
    except Exception as e:
        print(e)
        return None


def generate_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=128, n_fft=2048, hop_length=512)
    return mfcc


def generate_chroma(audio):
    chroma = librosa.feature.chroma_stft(y=audio, sr=22050, n_chroma=128, n_fft=2048, hop_length=512)

    # # Write the array to a text file with desired formatting
    # with open('chroma_spec.txt', 'w') as f:
    #     f.write('[')
    #     for row in chroma:
    #         f.write('[%s],' % ', '.join(map(str, row)))
    #     f.write(']')
    #
    # # Read the text file
    # with open('chroma_spec.txt', 'r') as f:
    #     # Read the content of the file
    #     content = f.read()
    #
    # # Remove the outer square brackets
    # content = content.strip('[]')
    #
    # # Split the content into individual row strings
    # rows = content.split('],[')
    #
    # # Convert each row string into a list of float32 values
    # data = []
    # for row in rows:
    #     # Remove any extra whitespace and closing bracket from the row string
    #     row = row.strip().strip(']')
    #     # Split the row string into individual values
    #     values = row.split(',')
    #     # Convert each non-empty value to float32 after removing non-numeric characters
    #     row_values = [np.float32(value.strip('][')) for value in values if value.strip('][')]
    #     data.append(row_values)
    #
    # # Convert the data into a NumPy array of float32 values
    # chroma_spec = np.array(data, dtype=np.float32)
    #
    # print("Mel spectrogram loaded from mel_spec.txt:")
    # print(chroma_spec)
    # if (chroma_spec == chroma).all():
    #     print("Both arrays are equal")
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
    for i in range(len(diseases)):
        severities.append(severity[diseases[i]])
    return severities


def convert_probabilities_to_float(probabilities):
    floated_probabilities = []
    for i in range(len(probabilities)):
        floated_probabilities.append(float(probabilities[i]))
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
        print("Secondary result: ", secondary_result)

        severities = get_severity_level(secondary_result['diseases'])

        secondary_result['severities'] = severities

        floated_probabilities = convert_probabilities_to_float(secondary_result['probabilities'])

        secondary_result['probabilities'] = floated_probabilities

        print("Secondary result: ", secondary_result)

        return secondary_result
