import joblib
from flask import Flask, request, jsonify
from Application.BackendServer.spectrogram import createSpectrogram
import random

app = Flask(__name__)

# model = joblib.load('tf_binary_model.h5')
# print("model loaded")

# @app.before_request()
# def load_model():
#     global model
#     model = joblib.load('tf_binary_model.h5')
#     print("model loaded")


@app.route('/')
def index():
    print("hello")
    return "hello"


@app.route('/giveResult', methods=['GET'])
def giveResult():
    print("hiiiii")
    audio = request.files['audio']
    result = createSpectrogram(audio)
    return result


if __name__ == '__main__':
    app.run(debug=True)
