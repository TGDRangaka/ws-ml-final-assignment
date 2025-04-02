# pip install Flask

# python --version

from flask import Flask
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load preprocessor and model
preprocessor = joblib.load("preprocessor.pkl")
svm_model = joblib.load("svm_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>AI v03</h1>'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80) # to change the port .