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
    print("hiiiii")
    audio = request.files['audio']
    createSpectrogram(audio)
    random_value = random.choice([0, 1])
    if random_value == 0:
        result_data = {'result': False, 'diseases': {}}
        return jsonify(result_data)
    else:
        result_data = {'result': True, 'diseases': {'disease1': 0.5, 'disease2': 0.3, 'disease3': 0.2}}
        return jsonify(result_data)


if __name__ == '__main__':
    app.run(debug=True)
