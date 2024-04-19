from spectrogram import generate_mel_spec, generate_mfcc, generate_chroma, expand_dimension, convert_probabilities_to_float, get_severity_level
from classificationModel import do_primary_prediction, do_secondary_prediction


# def test_convert_probabilities_to_float():
#     assert convert_probabilities_to_float([0.6320073, 0.20772938, 0.13448954]) == [0.6320073008537292, 0.20772938430309296, 0.1344895362854004]


def test_get_severity_level():
    assert get_severity_level(['URTI', 'Pleural Effusion', 'Asthma']) == [2, 3, 1]