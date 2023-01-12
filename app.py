from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics 
import scoring
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction")
def predict():
    data = request.args.get('filename')
    prediction = diagnostics.model_predictions(data)
    #call the prediction function you created in Step 3
    return str(prediction)
    #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():   
    f1score = scoring.score_model()  
    #check the score of the deployed model
    return str(f1score)
    #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():
    mean, median, std, na_perc = diagnostics.dataframe_summary()
    stats = [mean, median, std]
    #check means, medians, and modes for each column
    return str(stats)
    #return a list of all calculated summary statistics , median, std, na_perc

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnosticsstats():
    #na_perc = diagnostics.dataframe_summary()
    timing = diagnostics.execution_time()  
    outdated = diagnostics.outdated_packages_list() 
    na_perc = diagnostics.dataframe_summary()
    stats = [timing, na_perc, outdated]
    #check timing and percent NA values
    return str(stats)
    #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
