from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("bike_mileage_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    
    if request.method == "POST":
        cc = float(request.form["cc"])
        weight = float(request.form["weight"])
        input_data = np.array([[cc, weight]])  # 2 features
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run()  # NO debug=True for production
