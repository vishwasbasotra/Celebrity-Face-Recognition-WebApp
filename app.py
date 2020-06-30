import util
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config.update(
    dict(SECRET_KEY="powerful secretkey", WTF_CSRF_SECRET_KEY="a csrf secret key")
)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def predictions():
    image_data = request.form['image_data']

    response = jsonify(util.classify_image(image_data))

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    print('Starting Python Flask Server For Sports Celebrity Image Classification!!!')
    util.load_artifacts()
    app.run(debug=True)
