from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_model_path']) 
final_data_path = os.path.join(config['output_folder_path'])

score_record=open(f'./{dataset_csv_path}/latestscore.txt','w')

#################Function for model scoring
def score_model():
    df = pd.read_csv(f'./{final_data_path}/finaldata.csv')
    with open(f'./{dataset_csv_path}/trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    X = df.drop(['corporation', 'exited'], axis=1)
    y = df['exited']
    predicted = model.predict(X)
    f1score = metrics.f1_score(predicted,y)
    score_record.write(str(f1score))
    return f1score
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

if __name__ == '__main__':
    score_model()