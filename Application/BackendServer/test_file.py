from spectrogram import generate_mel_spec, generate_mfcc, generate_chroma, expand_dimension, \
    convert_probabilities_to_float, get_severity_level
from classificationModel import do_primary_prediction, do_secondary_prediction
import numpy as np
import librosa


# def test_convert_probabilities_to_float():
#     assert convert_probabilities_to_float([0.6320073, 0.20772938, 0.13448954]) == [0.6320073008537292, 0.20772938430309296, 0.1344895362854004]


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