from flask import Flask, render_template, request
import numpy as np
import pickle
from gevent.pywsgi import WSGIServer


app = Flask(__name__)
model = pickle.load(open('lung.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        GENDER= int(request.form['GENDER'])
        AGE = int(request.form['AGE'])
        SMOKING= int(request.form['SMOKING'])
        YELLOW_FINGERS = int(request.form['YELLOW_FINGERS'])
        ANXIETY = int(request.form['ANXIETY'])
        PEER_PRESSURE = int(request.form['PEER_PRESSURE'])
        CHRONIC_DISEASE = int(request.form['CHRONIC_DISEASE'])
        FATIGUE = int(request.form['FATIGUE'])
        ALLERGY = int(request.form['ALLERGY'])
        WHEEZING = int(request.form['WHEEZING'])
        ALCOHOL_CONSUMING = int(request.form['ALCOHOL_CONSUMING'])
        COUGHING = int(request.form['COUGHING'])
        SHORTNESS_OF_BREATH = int(request.form['SHORTNESS_OF_BREATH'])
        SWALLOWING_DIFFICULTY = int(request.form['SWALLOWING_DIFFICULTY'])
        CHEST_PAIN = int(request.form['CHEST_PAIN'])
        LUNG_CANCER = int(request.form['LUNG_CANCER'])\

        values = np.array([[GENDER,AGE,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,FATIGUE,ALLERGY,WHEEZING,ALCOHOL_CONSUMING,COUGHING,SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN,LUNG_CANCER]])
        prediction = model.predict(values)

        return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)

