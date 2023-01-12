
import os
import sys
import json
import subprocess
import pickle
import pandas as pd
from sklearn import metrics

with open('config.json','r') as f:
    config = json.load(f) 

deployment_directory = os.path.join(config['prod_deployment_path'])
ingestion_directory = os.path.join(config['output_folder_path']) 
model_directory = os.path.join(config['output_model_path']) 
inputs = os.path.join(config['input_folder_path']) 

##################Check and read new data
#first, read ingestedfiles.txt
with open(f'./{deployment_directory}/ingestedfiles.txt', 'r') as file:
    ingest = file.read()
    ingest = ingest.split()
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
files = []
for file in os.listdir(f'./{inputs}'):
    files.append(file)

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if ingest != files:
    subprocess.run(['python', 'ingestion.py'])
else:
    sys.exit()


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(f'./{model_directory}/latestscore.txt', 'r') as file:
    latest_score = file.read()
    latest_score = float(latest_score)

with open(f'./{deployment_directory}/trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)

df = pd.read_csv(f'./{ingestion_directory}/finaldata.csv')

X = df.drop(['corporation', 'exited'], axis=1)
y = df['exited']
predicted = model.predict(X)
current_score = metrics.f1_score(predicted,y)
current_score = float(current_score)

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if latest_score < current_score:
    sys.exit()
else:
    pass


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
subprocess.run(['python', 'training.py'])
subprocess.run(['python', 'deployment.py'])


##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
subprocess.run(['python', 'diagnostics.py'])
subprocess.run(['python', 'reporting.py'])
subprocess.run(['python', 'apicalls.py'])




