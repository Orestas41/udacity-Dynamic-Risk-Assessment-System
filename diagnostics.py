
import pandas as pd
import numpy as np
import pickle
import timeit
import os
import json
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['prod_deployment_path'])
final_data_path = os.path.join(config['output_folder_path']) 

##################Function to get model predictions
def model_predictions(data):
    #read the deployed model and a test dataset, calculate predictions
    with open(f'./{dataset_csv_path}/trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    df = pd.read_csv(data)
    X = df.drop(['corporation', 'exited'], axis=1)
    predicted = model.predict(X)
    return predicted #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    df = pd.read_csv(f'./{final_data_path}/finaldata.csv')
    num = df.drop(['corporation'], axis=1)
    mean = num.mean()
    median = num.median()
    std = num.std()
    na = num.isna().sum()
    na_perc = (na / num.count()) * 100
    #calculate summary statistics here
    return mean, median, std, na_perc
    #return value should be a list containing all summary statistics

##################Function to get timings
def execution_time():
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_timing=timeit.default_timer() - starttime

    starttime = timeit.default_timer()
    os.system('python3 training.py')
    training_timing=timeit.default_timer() - starttime

    timings = [ingestion_timing, training_timing]
    #calculate timing of training.py and ingestion.py
    return timings
    #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    return outdated