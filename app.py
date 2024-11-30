import pandas as pd
from flask import Flask, request, render_template
import joblib

mymodel = joblib.load("model_test.joblib")

app = Flask(__name__) # __name__ = app
model_class = {
    "0":"setosa",
    "1":"versicolor",
    "2":"virginica"
}

@app.route("/", methods = ["GET", "POST"])

def index():

    if request.method == "POST":

        val1 = float(request.form["val1"])
        val2 = float(request.form["val2"])
        val3 = float(request.form["val3"])
        val4 = float(request.form["val4"])

        my_df = pd.DataFrame([[val1, val2, val3, val4]],columns=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'])
        prediction = str(mymodel.predict(my_df)[0])
        pred_class = model_class[prediction]

    else:

        pred_class = None

    return render_template("index.html", prediction = pred_class)

