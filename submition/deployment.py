from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = config['output_folder_path']
dataset_csv_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

files_record_deploy=open(f'./{prod_deployment_path}/ingestedfiles.txt','w')

####################function for deployment
def store_model_into_pickle():

    with open(f'./{dataset_csv_path}/trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    pickle.dump(model, open(f'./{prod_deployment_path}/trainedmodel.pkl', 'wb'))

    with open(f'./{dataset_csv_path}/latestscore.txt','r') as score1, open(f'./{prod_deployment_path}/latestscore.txt','w') as score2:
        for line in score1:
             score2.write(line)

    with open(f'./{output_folder_path}/ingestedfiles.txt','r') as ingest1, open(f'./{prod_deployment_path}/ingestedfiles.txt','w') as ingest2:
        for line in ingest1:
             ingest2.write(line)
    #copy the latest pickle file, the latestscore.txt value, and the ingestedfiles.txt file into the deployment directory
        
if __name__ == '__main__':
    store_model_into_pickle()       
        

