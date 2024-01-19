import joblib
from flask import Flask, request, jsonify
from spectrogram import createSpectrogram
import random

app = Flask(__name__)


@app.route('/')
def index():
    print("hello")
    return "hello"


@app.route('/giveResult', methods=['POST'])
def giveResult():
    audio = request.data

    with open('received_audio.wav', 'wb') as wav_file:
        wav_file.write(audio)

    result = createSpectrogram(audio)
    print("hello")
    return "hello"


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0', port=8080)
