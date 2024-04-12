
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
    # fileName = audio.filename
    # print(fileName)
    with open('received_audio.wav', 'wb') as wav_file:
        wav_file.write(audio)

    result = createSpectrogram('received_audio.wav')
    print("hello")
    return result


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)

