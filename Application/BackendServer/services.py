from flask import Flask, request, jsonify
from spectrogram import generateResult
import random
import librosa

app = Flask(__name__)


@app.route('/')
def index():
    return "hello"


@app.route('/giveResult', methods=['POST'])
def giveResult():

    if 'audio' not in request.files:
        return 'No audio file in the request', 400

    audio = request.files['audio']

    result = generateResult(audio)

    if result is None:
        return jsonify({'result': "Error occurred while generating spectrograms", 'diseases': {}})

    return result


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
