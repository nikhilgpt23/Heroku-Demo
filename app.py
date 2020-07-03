# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 09:11:58 2020

@author: NIKHIL
"""

import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods= ['POST'])

def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template('index.html',
        prediction_text = 'Employee salary should be $ {}'.format(output))
                           
if __name__ == "_main_":
    app.run(debug = True)
