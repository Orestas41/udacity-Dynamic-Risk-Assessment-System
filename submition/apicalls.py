import subprocess
import os
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"

    
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_model_path']) 

#Call each API endpoint and store the responses
response1 = subprocess.run(['curl', '127.0.0.1:8000/prediction?filename=./sourcedata/finaldata.csv'],capture_output=True).stdout
response2 = subprocess.run(['curl', '127.0.0.1:8000/scoring'],capture_output=True).stdout
response3 = subprocess.run(['curl', '127.0.0.1:8000/summarystats'],capture_output=True).stdout
response4 = subprocess.run(['curl', '127.0.0.1:8000/diagnostics'],capture_output=True).stdout

responses = [response1, response2, response3, response4]

open(f'./{dataset_csv_path}/apireturns2.txt', 'w').close()

for each_call in responses:
    decoded_call = each_call.decode('utf-8')
    
    api_returns=open(f'./{dataset_csv_path}/apireturns2.txt','w')

    api_returns.write(decoded_call)

    api_returns.close()

