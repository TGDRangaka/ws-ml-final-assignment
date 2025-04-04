# pip install Flask

# python --version

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Load preprocessor and model
preprocessor = joblib.load("preprocessor.pkl")
svm_model = joblib.load("svm_model.pkl")

app = Flask(__name__)

# # Define categorical and numerical columns
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                        'contact', 'month', 'day_of_week', 'poutcome']
numerical_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                      'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']


@app.route('/form')
def index():
    return render_template('bank-form.html')

@app.route('/predict', methods=['POST'])
def customer_details():
    form = request.form
    manual_input = pd.DataFrame([{
        'age': form['age'], 'campaign': np.nan, 'pdays': np.nan, 'previous': np.nan,
        'emp.var.rate': np.nan, 'cons.price.idx': np.nan, 'cons.conf.idx': np.nan,
        'euribor3m': np.nan, 'nr.employed': np.nan,
        'job': form['job'], 'marital': form['marital'], 'education': form['education'], 'default': form['default'],
        'housing': form['housing'], 'loan': form['loan'], 'contact': np.nan, 'month': np.nan,
        'day_of_week': np.nan, 'poutcome': np.nan
    }])

    manual_transformed = preprocessor.transform(manual_input)
    prediction = svm_model.predict(manual_transformed)
    
    result = 'No'
    if(prediction[0] == 1):
        result = 'Yes'
    
    # return the predicted output as a response
    return render_template('dashboard.html', predict=result)


@app.route('/')
def home():
    return '<h1>Term Deposit AI v01</h1>'

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=80) # to change the port .