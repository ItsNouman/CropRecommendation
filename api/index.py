from flask import Flask, request, render_template
import numpy as np
import pickle
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
sc = pickle.load(open(os.path.join(BASE_DIR, "standardscaler.pkl"), "rb"))
mx = pickle.load(open(os.path.join(BASE_DIR, "minmaxscaler.pkl"), "rb"))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
        6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
        10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean",
        18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
        21: "Chickpea", 22: "Coffee"
    }

    result = crop_dict.get(prediction[0], "Sorry, could not determine best crop.")
    return render_template('index.html', result=f"{result} is the best crop to be cultivated right there")

if __name__ == "__main__":
    app.run()
