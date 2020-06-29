import numpy as np
import json
import util
from flask import Flask, request, jsonify, url_for, redirect, render_template

app = Flask(__name__)


@app.route('/home', methods=['GET', 'POST'])
def classify_image():
    image_data = request.form['image_data']

    result = jsonify(util.classify_image(image_data))

    result.headers.add('Access-Control-Allow-Origin', '*')

    return result


if __name__ == "__main__":
    print('Starting Python Flask Server For Sports Celebrity Image Classification!!!')
    util.load_artifacts()
    app.run(debug=True)
