from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("bike_mileage_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        cc = float(request.form["cc"])
        weight = float(request.form["weight"])
        fuel = request.form["fuel"]

        fuel_type = 1 if fuel == "Petrol" else 0

        input_data = np.array([[cc, weight, fuel_type]])
        prediction = model.predict(input_data)[0]

        prediction = round(prediction, 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run()
