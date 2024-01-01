import joblib
from flask import Flask, request, jsonify
from spectrogram import createSpectrogram
import random

app = Flask(__name__)


@app.route('/')
def index():
    print("hello")
    return "hello"


@app.route('/giveResult', methods=['GET'])
def giveResult():
    audio = request.files['audio']
    result = createSpectrogram(audio)
    return result


if __name__ == '__main__':
    app.run(debug=True)
