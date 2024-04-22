import logging
from unittest.mock import Mock

from flask import jsonify, Flask, request

from spectrogram import generate_mel_spec, generate_mfcc, generate_chroma, expand_dimension, \
    convert_probabilities_to_float, get_severity_level, createSpectrogram
from classificationModel import do_primary_prediction, do_secondary_prediction, get_disease_names, get_top_3_diseases
import numpy as np
import librosa
import pytest

# Create a Flask application instance
app = Flask(__name__)


def test_get_severity_level():
    assert get_severity_level(['URTI', 'Pleural Effusion', 'Asthma']) == [2, 3, 1]


def test_generate_mel_spec():
    audio = '1.wav'
    # Load audio file
    y, sr = librosa.load(audio, sr=22500, duration=6)
    # Read the text file
    with open('mel_spec.txt', 'r') as f:
        # Read the content of the file
        content = f.read()

    # Remove the outer square brackets
    content = content.strip('[]')

    # Split the content into individual row strings
    rows = content.split('],[')

    # Convert each row string into a list of float32 values
    data = []
    for row in rows:
        # Remove any extra whitespace and closing bracket from the row string
        row = row.strip().strip(']')
        # Split the row string into individual values
        values = row.split(',')
        # Convert each non-empty value to float32 after removing non-numeric characters
        row_values = [np.float32(value.strip('][')) for value in values if value.strip('][')]
        data.append(row_values)

    # Convert the data into a NumPy array of float32 values
    mel_spec = np.array(data, dtype=np.float32)

    assert (generate_mel_spec(y) == mel_spec).all()


def test_generate_mel_exception():
    audio = []
    assert generate_mel_spec(audio) is None


def test_generate_mfcc():
    audio = '1.wav'
    # Load audio file
    y, sr = librosa.load(audio, sr=22500, duration=6)
    # Read the text file
    with open('mfcc_spec.txt', 'r') as f:
        # Read the content of the file
        content = f.read()

    # Remove the outer square brackets
    content = content.strip('[]')

    # Split the content into individual row strings
    rows = content.split('],[')

    # Convert each row string into a list of float32 values
    data = []
    for row in rows:
        # Remove any extra whitespace and closing bracket from the row string
        row = row.strip().strip(']')
        # Split the row string into individual values
        values = row.split(',')
        # Convert each non-empty value to float32 after removing non-numeric characters
        row_values = [np.float32(value.strip('][')) for value in values if value.strip('][')]
        data.append(row_values)

    # Convert the data into a NumPy array of float32 values
    mfcc = np.array(data, dtype=np.float32)

    assert (generate_mfcc(y) == mfcc).all()


def test_generate_chroma():
    audio = '1.wav'
    # Load audio file
    y, sr = librosa.load(audio, sr=22500, duration=6)
    # Read the text file
    with open('chroma_spec.txt', 'r') as f:
        # Read the content of the file
        content = f.read()

    # Remove the outer square brackets
    content = content.strip('[]')

    # Split the content into individual row strings
    rows = content.split('],[')

    # Convert each row string into a list of float32 values
    data = []
    for row in rows:
        # Remove any extra whitespace and closing bracket from the row string
        row = row.strip().strip(']')
        # Split the row string into individual values
        values = row.split(',')
        # Convert each non-empty value to float32 after removing non-numeric characters
        row_values = [np.float32(value.strip('][')) for value in values if value.strip('][')]
        data.append(row_values)

    # Convert the data into a NumPy array of float32 values
    chroma = np.array(data, dtype=np.float32)

    assert (generate_chroma(y) == chroma).all()


def test_convert_probabilities_to_float():
    probabilities = [0.1, 0.2, 0.3, 0.4]
    converted = convert_probabilities_to_float([0.1, 0.2, 0.3, 0.4])
    assert convert_probabilities_to_float(probabilities) == [0.1, 0.2, 0.3, 0.4]


def test_get_disease_names():
    indexes = [8, 6, 0]
    assert get_disease_names(indexes) == ['URTI', 'Pleural Effusion', 'Asthma']


def test_get_top_3_diseases():
    result = [[1.3448954e-01, 2.3922540e-04, 3.5269838e-03, 7.7271527e-03, 1.1881822e-03, 1.0382760e-02, 2.0772938e-01,
               2.7094933e-03, 6.3200730e-01]]
    print("top 3 diseases by test", get_top_3_diseases(result))
    assert get_top_3_diseases(result) == ([8, 6, 0], [0.6320073, 0.20772938, 0.13448954])


def test_expand_dimension():
    input_array = np.array([[[1, 2, 3], [4, 5, 6]]])
    output = np.array([[[[1, 2, 3], [4, 5, 6]]]])
    print("output shape", output.shape)
    assert (expand_dimension(input_array) == output).all()


def test_expand_dimension_negative():
    input_array = np.array([[1, 2, 3], [4, 5, 6]])
    assert expand_dimension(input_array) is None


def test_create_spectrogram_none(mocker):
    audio = '1.wav'
    mocker_response_expand = mocker.Mock()
    mocker_response_expand.return_value = None
    mocker.patch('spectrogram.expand_dimension', return_value=mocker_response_expand.return_value)

    print(mocker_response_expand)
    mocker_response_primary = mocker.Mock()
    mocker_response_primary.return_value = True
    mocker.patch('spectrogram.do_primary_prediction', return_value=mocker_response_primary.return_value)

    assert createSpectrogram(audio) is None


def test_create_spectrogram_healthy(mocker):
    audio = '1.wav'
    mocker_response_expand = mocker.Mock()
    mocker_response_expand.return_value = np.array([[[[1, 2, 3], [4, 5, 6]]]])
    mocker.patch('spectrogram.expand_dimension', return_value=mocker_response_expand.return_value)

    mocker_response_primary = mocker.Mock()
    mocker_response_primary.return_value = False
    mocker.patch('spectrogram.do_primary_prediction', return_value=mocker_response_primary.return_value)

    mocker_response_secondary = mocker.Mock()
    mocker_response_secondary.return_value = {'diseases': ['URTI', 'Pleural Effusion', 'Asthma'],
                                              'probabilities': [0.6320073, 0.20772938, 0.13448954]}
    mocker.patch('spectrogram.do_secondary_prediction', return_value=mocker_response_secondary.return_value)

    # Set up application context using app.test_request_context()
    with app.test_request_context():
        ExpectedResult = b'{"diseases":{},"result":false}\n'
        ActualResult = createSpectrogram(audio).data
        assert ActualResult == ExpectedResult


def test_create_spectrogram_unhealthy(mocker):
    audio = '1.wav'
    mocker_response_expand = mocker.Mock()
    mocker_response_expand.return_value = np.array([[[[1, 2, 3], [4, 5, 6]]]])
    mocker.patch('spectrogram.expand_dimension', return_value=mocker_response_expand.return_value)

    mocker_response_primary = mocker.Mock()
    mocker_response_primary.return_value = True
    mocker.patch('spectrogram.do_primary_prediction', return_value=mocker_response_primary.return_value)

    mocker_response_secondary = mocker.Mock()
    mocker_response_secondary.return_value = {'diseases': ['URTI', 'Pleural Effusion', 'Asthma'],
                                              'probabilities': [0.6320073, 0.20772938, 0.13448954]}
    mocker.patch('spectrogram.do_secondary_prediction', return_value=mocker_response_secondary.return_value)

    # Set up application context using app.test_request_context()
    with app.test_request_context():
        ExpectedResult = {'diseases': ['URTI', 'Pleural Effusion', 'Asthma'],
                          'probabilities': [0.6320073, 0.20772938, 0.13448954], 'severities': [2, 3, 1]}
        ActualResult = createSpectrogram(audio)
        assert ActualResult == ExpectedResult
