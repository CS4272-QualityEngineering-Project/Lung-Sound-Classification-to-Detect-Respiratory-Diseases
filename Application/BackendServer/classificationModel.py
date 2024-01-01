from keras.models import load_model


def do_primary_prediction(spectrogram):
    loaded_model = load_model('binary_model.h5')
    print("model loaded")
    result = loaded_model.predict(spectrogram, batch_size=1)
    print("result", result[0][0])
    if result[0][0] > 0.5:
        return False
    else:
        return True


def do_secondary_prediction(spectrogram):
    loaded_model = load_model('disease_catego_model.h5')
    print("secondary model loaded")
    result = loaded_model.predict(spectrogram)
    print("secondary result", result)
    # Sort the array in descending order
    sorted_numbers = sorted(result[0], reverse=True)
    # Get the top 3 values
    top3_values = sorted_numbers[:3]
    # Get the indexes of the top 3 values
    indexes_of_top3_values = sorted(range(len(result[0])), key=lambda i: result[0][i], reverse=True)[:3]
    print("sorted_numbers", top3_values)
    print("indexes of top 3 results", indexes_of_top3_values)
    disease_list = ['Asthma', 'Bronchiectasis', 'Bronchiolitis', "Bronchitis", 'COPD', 'Lung Fibrosis',
                    'Pleural Effusion', 'Pneumonia', 'URTI']
    predicted_disease = {}
    for i in range(3):
        print(i)
        disease = disease_list[indexes_of_top3_values[i]]
        print("disease", disease)
        predicted_disease[disease] = top3_values[i]
    print(predicted_disease)
    return predicted_disease
