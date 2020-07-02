import cv2
import pywt
import pickle
import numpy as np
import base64
import json
from flask import Flask, request, jsonify, render_template, redirect

app = Flask(__name__)
app.config.update(
    dict(SECRET_KEY="powerful secretkey", WTF_CSRF_SECRET_KEY="a csrf secret key")
)

__class_name_to_number = {}
__class_number_to_name = {}

__model = pickle.load(open("final_model.pickle", "rb"))


def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []

    for img in imgs:
        scaled_raw_img = cv2.resize(img, (32, 32))
        img_hr = w2d(img, 'db1', 5)
        scaled_hr_img = cv2.resize(img_hr, (32, 32))
        combined_img = np.vstack((scaled_raw_img.reshape(32 * 32 * 3, 1), scaled_hr_img.reshape(32 * 32, 1)))

        len_image_array = (32 * 32 * 3) + (32 * 32)

        final = combined_img.reshape(1, len_image_array).astype(float)

        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.round(__model.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result


def class_number_to_name(class_num):
    if class_num == 0:
        player = 'lionel_messi'

    elif class_num == 1:
        player = 'maria_sharapova'

    elif class_num == 2:
        player = 'roger_federer'

    elif class_num == 3:
        player = 'serena_williams'

    elif class_num == 4:
        player = 'virat_kohli'

    return player


def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./model/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./model/opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces


def load_artifacts():
    print("Loading saved artifacts...Start")
    global __class_name_to_number
    global __class_number_to_name
    global __model

    with open('class_dictionary.json', 'r') as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    print("Loading save artifacts...done!")


def w2d(img, mode='haar', level=1):
    imArray = img

    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)

    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255

    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level)

    # process coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/prediction", methods=["POST"])
def prediction():

    if request.method == 'POST':

        image_data = request.form['b64']
        result = classify_image(image_data)
        print(result)

        if len(result) == 0:
            message = "Can't classify image. Classifier was not able to detect face and two eyes properly"
            return render_template("prediction.html", message=message, messi="NA", sharapova="NA", federer="NA",
                                   serena="NA", virat="NA")

        match = None
        bestScore = -1

        for i in range(len(result)):
            maxScoreForThisClass = max(result[i]['class_probability'])

            if maxScoreForThisClass > bestScore:
                match = result[i]
                bestScore = maxScoreForThisClass

        playerName = match['class']

        player = playerName.split('_')
        player = " ".join(player).upper()

        messi = match['class_probability'][0]
        sharapova = match['class_probability'][1]
        federer = match['class_probability'][2]
        serena = match['class_probability'][3]
        virat = match['class_probability'][4]

        return render_template('prediction.html',player=player, playerName=playerName, messi=messi, sharapova=sharapova,
                               federer=federer, serena=serena, virat=virat)


if __name__ == "__main__":
    print('Starting Python Flask Server For Sports Celebrity Image Classification!!!')
    load_artifacts()
    app.run(debug=True)
