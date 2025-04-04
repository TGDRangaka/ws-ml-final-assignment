# pip install Flask

# python --version
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


lr_model = joblib.load("lr_model.joblib")
svm_model = joblib.load("svm_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")

categorical_features = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
numerical_features = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]


def prepare_input(data):
    return pd.DataFrame(
        [
            {
                "age": int(data["age"]),
                "job": data["job"],
                "marital": data["marital"],
                "education": data["education"],
                "default": data["default"],
                "housing": data["housing"],
                "loan": data["loan"],
                "contact": "cellular",
                "month": "may",
                "day_of_week": "mon",
                "campaign": 1,
                "pdays": 0,
                "previous": 0,
                "duration": int(data.get("duration", 200)),
                "poutcome": "nonexistent",
                "emp.var.rate": 1.1,
                "cons.price.idx": 93.994,
                "cons.conf.idx": -36.1,
                "euribor3m": 4.857,
                "nr.employed": 5191.0,
            }
        ]
    )

def getPredict(val):
    if(val == 1): return 'Yes'
    return 'No'

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/svm")
def svm_page():
    return render_template("svmModel.html")


@app.route("/lrm")
def lrm_page():
    return render_template("lrModel.html")


@app.route("/svm-predict", methods=["POST"])
def predict_svm():
    data = request.form

    input_df = prepare_input(data)

    manual_transformed = preprocessor.transform(input_df)
    prediction = svm_model.predict(manual_transformed)

    return render_template("predict.html", predict=getPredict(prediction[0]))


@app.route("/lrm-predict", methods=["POST"])
def predict_lrm():
    data = request.form

    input_df = prepare_input(data)

    manual_transformed = preprocessor.transform(input_df)
    prediction = lr_model.predict(manual_transformed)

    return render_template("predict.html", predict=getPredict(prediction[0]))


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=80)  # to change the port .