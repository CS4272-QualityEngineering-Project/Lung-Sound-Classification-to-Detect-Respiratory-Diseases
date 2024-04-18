from keras.models import load_model


def do_primary_prediction(spectrogram):
    loaded_model = load_model('binary_model.h5')
    result = loaded_model.predict(spectrogram, batch_size=1)
    if result[0][0] > 0.5:
        return False
    else:
        return True


def get_top_3_diseases(result):
    # Sort the array in descending order
    sorted_numbers = sorted(result[0], reverse=True)
    # Get the top 3 values
    top3_values = sorted_numbers[:3]
    # Get the indexes of the top 3 values
    indexes_of_top3_values = sorted(range(len(result[0])), key=lambda i: result[0][i], reverse=True)[:3]
    return indexes_of_top3_values, top3_values


def get_disease_names(indexes_of_top3_values):
    disease_list = ['Asthma', 'Bronchiectasis', 'Bronchiolitis', "Bronchitis", 'COPD', 'Lung Fibrosis',
                    'Pleural Effusion', 'Pneumonia', 'URTI']
    disease_names = []
    for i in range(3):
        disease = disease_list[indexes_of_top3_values[i]]
        disease_names.insert(i, disease)
    return disease_names


def do_secondary_prediction(spectrogram):
    loaded_model = load_model('disease_catego_model.h5')
    result = loaded_model.predict(spectrogram)

    indexes_of_top3_values, top3_values = get_top_3_diseases(result)

    diseases = get_disease_names(indexes_of_top3_values)

    predicted_disease = {'diseases': diseases, 'probabilities': top3_values}
    return predicted_disease
