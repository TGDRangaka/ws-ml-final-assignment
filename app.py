from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load preprocessor and model
preprocessor = joblib.load("preprocessor.pkl")
svm_model = joblib.load("svm_model.pkl")

app = Flask(__name__)

# Define categorical and numerical columns
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                        'contact', 'month', 'day_of_week', 'poutcome']
numerical_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                      'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']


@app.route('/')
def index():
    return render_template('bank-form.html')

@app.route('/predict', methods=['POST'])
def customer_details():
    form = request.form
    manual_input = pd.DataFrame([{
        'age': form['age'], 'duration': np.nan, 'campaign': np.nan, 'pdays': np.nan, 'previous': np.nan,
        'emp.var.rate': np.nan, 'cons.price.idx': np.nan, 'cons.conf.idx': np.nan,
        'euribor3m': np.nan, 'nr.employed': np.nan,
        'job': form['job'], 'marital': form['marital'], 'education': form['education'], 'default': form['default'],
        'housing': form['housing'], 'loan': form['loan'], 'contact': np.nan, 'month': np.nan,
        'day_of_week': np.nan, 'poutcome': np.nan
    }])

    manual_input[numerical_features] = manual_input[numerical_features].fillna(0)
    manual_input[categorical_features] = manual_input[categorical_features].fillna('unknown')

    manual_transformed = preprocessor.transform(manual_input)
    prediction = svm_model.predict(manual_transformed)
    
    # return the predicted output as a response
    return render_template('dashboard.html', predict=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=80) # to change the port .