from flask import Flask, request, render_template
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load model & scalers from same folder
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
sc = pickle.load(open(os.path.join(BASE_DIR, "standardscaler.pkl"), "rb"))
mx = pickle.load(open(os.path.join(BASE_DIR, "minmaxscaler.pkl"), "rb"))

# explicit template/static folders (they are inside api/)
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        feature_list = [
            float(request.form['Nitrogen']),
            float(request.form['Phosporus']),   # keep same name as your form
            float(request.form['Potassium']),
            float(request.form['Temperature']),
            float(request.form['Humidity']),
            float(request.form['pH']),
            float(request.form['Rainfall'])
        ]
    except Exception as e:
        return render_template("index.html", result=f"Invalid input: {e}")

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

    result = crop_dict.get(int(prediction[0]), "Sorry, could not determine best crop.")
    return render_template('index.html', result=f"{result} is the best crop to be cultivated right there")

if __name__ == "__main__":
    # keep this for local testing
    app.run(debug=True, port=5000)
